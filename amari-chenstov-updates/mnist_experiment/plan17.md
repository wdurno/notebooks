# Implementation Plan 17: Observation-Clock Adaptive Pi

**Motivation:** obtain useful continual learning per scarce observation and per
unit of compute, while retaining previously learned predictive capability. A fixed
new-information weight can be too aggressive during a long quiet interval and
too conservative during genuine change. This study asks whether a causal
state-space observer can use that distinction to improve digit-9 acquisition
and digit-0--8 retention, not merely produce smoother recommendations.

Earlier optimal-$\pi$ studies found severe inflation of the inferred movement
energy and little demonstrated predictive value. Their accepted-parameter
trends could mix population movement, learner catch-up, optimizer behavior, and
noise. Some historical observers also used environmental distance as their
clock, so exact holds did not continue their statistical averaging. Common-
anchor score observations and an observation-clock filter address different
parts of that problem; both need explicit controls.

> **Question:** does Kalman-informed estimation of movement and uncertainty
> improve the acquisition/retention trade-off compared with a tuned fixed action,
> covariance-only adaptation, and a simpler common-anchor EMA observer?

**Status:** Phases 0--4 complete, authorized and finished on 2026-11-01
(Pacific time). **Phase 5 execution authorized on 2026-11-02; acceptance passed
and Phase 5A--D have completed in `retention_v1`: 120 development and 48 fresh
paths, with all 205 scientific units verified.**
The separately approved Phase 5E zero-ridge confirmation also completed in
`retention_supplement_v1`: eight paired blocks / 24 paths. **Phases 0--5 are
complete; their family compute is approximately 14.32 of 20 hours. Phase 6,
single-digit registration under continuous rotation drift, completed all
sub-phases on 2026-11-04 using approximately 4.30 of its separate 20 hours.
Phase 7, discrepancy-informed adaptation during sustained registration drift,
is authorized for implementation and execution on 2026-11-05.**
The original phases finished, including eight changing-schedule and four stationary
fresh replicas across four policies: **48 fresh trajectories**, without
significance stopping or extending the cohort. The completed root is
`cache/mnist_experiment/plan17/production_v4`. The original Phases 0--4 charged
about **4.14 hours**, including earlier revisions, acceptance, and closeout.
All 102 sealed units passed full-file checksum and pairing checks. Bounded
development repairs, results, and final artifact paths are recorded under
Execution Closeout below and in `PLAN17_HANDOFF.md`. No Plan 16 scientific
artifacts or sources were modified.
**Roles:** Astra supplies this design; Sol implements and executes all phases
after authorization, with authority to correct defects and keep useful work
running without further model switches or recurring user approvals.
**Compute:** **20 cumulative active wall-clock hours for Phases 0--5**, separately
authorized from Plan 16. Phase 6 has an additional, separately authorized
allocation of at most 20 hours as of 2026-11-04. Preserve the historical ledger.
Phase 7 has a separately authorized 20-active-hour ceiling as of 2026-11-05;
unused earlier allowances are not transferred or reset.
This is not permission to run plans consecutively.
**Report:** **`mnist_experiment/plan17_results.ipynb`**, created during Phase 0
and updated as work progresses. Do not append this study to Plan 15's notebook.

## Exploration, Not A Win Gate

The user does not want a predeclared numerical success criterion. Explore the
two accuracy outcomes and costs without requiring a significance threshold,
minimum improvement, or retention margin to continue or conclude. Improvement
in both accuracies is welcome; a gain in one, loss in another, uncertain result,
or informative null result must remain visible rather than be relabeled to fit
a binary verdict.

Measurement definitions, paired seeds, versioned protocols, and declared cohort
allocations still matter. Separate development from fresh replication, publish
nominal intervals and all planned contrasts, and label outcome-informed changes.
Interim notebook inspection is expected. Do not stop for significance, extend
because significance is close, or interpret nonsignificance as equivalence.

## Shared Substrate And Separation From Plan 16

Reuse [Plan 16](plan16.md)'s gauge-fixed 487-coordinate CNN, ordinary categorical
likelihood, diagonal initialization/Fisher observations, diagonal EWC plus
common ridge, optimizer, dataset isolation, regime generator, and predictive
metric definitions. Keep all coordinates trainable. Its primary learner has
no STM, replay training, or FD-KL penalty; this is deliberate isolation of the
action-policy question rather than a reproduction of Plan 15's architecture.
Do not import Plan 13's class-weighted objective into the fixed-batch formulas.

Plan 17 does **not** require Plan 16 to have run or succeeded. Shared source and
compatible initial assets can be reused only with explicit dependency hashes
and accounting; fresh-replication streams and seeds are independent. No Plan
16 outcome selects a Plan 17 action on a live trajectory. All Fisher estimates
remain diagonal; every Plan 17 learner uses **direct EMA, without an LFU**.
Do not cross Fisher-correction variants with action policies in this allocation.

Use the same starting stream: $m=4$, $p_{\rm low}=.005$, $p_{\rm high}=.10$,
three 352-arrival holds separated by two 128-arrival half-cosine ramps. Stationary
low-prevalence trajectories form a smaller negative-control cohort. Keep
within-class image distributions and the relative frequencies of digits 0--8
fixed. Save exact schedules and expected/realized 9 counts. Environment $p_t$
and segment boundaries are never controller inputs.

The five-half-life lower bound on hold duration uses the slowest retained fixed
control, not environmental distance. Also check filter settling and anchor-
refresh timescales. If these require longer holds, revise schedules for every
paired arm, reducing replica count before truncating a regime. Fresh replication
may vary hold/ramp lengths by pre-generated factors in $\{1,1.25\}$ while
respecting that lower bound. Record the factors before learning; all arms share
them. Varying lengths is a robustness block, not independent time-point replication.

Digit-9 mixtures are first. Continuous-drift or rotated-digit studies are
fallbacks only after reporting the initial evidence, under new explicit
protocol identities and the remaining budget. No automatic second dataset.

## Mathematical Contract

### Likelihood, Action, And Weight Concentration

Use $g=-s$ and the single model Fisher
$\mathcal I(\theta)=\mathbb E_\theta[s s^T]$. Its maintained representation is
$D_t=\operatorname{diag}(\widehat d_t)$. Diagonal approximations, covariance
estimates for an observer, and the true Fisher are different objects; do not
rename an observer covariance as another Fisher estimand.

Accumulate Fisher vectors, score moments, and filter calculations in float64;
record and validate the precision used by the underlying model derivative
passes. Any numerical precision change is common to affected paired arms.

Use the applied **fixed-batch** risk model, not the Bernoulli-composition
variance law. Track the exact concentration of the algorithm's nominal
observation weights:

$$
q_{t+1}=(1-\pi_t)^2q_t+\frac{\pi_t^2}{m_t},\qquad q_0=1/n_0.
$$

Here $n_0$ is the shared initialization sample count. This recursion is exact
for the declared weights; interpreting $q_t$ as estimator covariance requires
additional local calibration. Repeated optimizer passes do not create new
observations, and repeated image IDs do not imply independent information.
Record arrivals and unique IDs separately.

Before observing $B_t$, choose and freeze $\pi_t$ from past information. Use it
in both the direct Fisher recursion and the EWC odds:

$$
\widehat d_t=(1-\pi_t)\widehat d_{t-1}+\pi_t z_{B_t}(\theta_t),
\qquad \omega_t=\frac{1-\pi_t}{\pi_t}.
$$

The learner minimizes the same objective as Plan 16 with $c$ replaced by
$\pi_t$, anchored at $\theta_t$. The first arrival uses the shared initial
diagonal, with unavailable prior measurements handled explicitly. No policy
may use $B_t$ to choose the action that weights $B_t$. Observer updates from
$B_t$ become eligible only for the next action. Persist this ordering in tests.

With a movement-energy estimate $S_t$ and a one-observation covariance-risk
scale $V_t$, the common surrogate recommendation is

$$
\pi_t^{\rm rec}=
\frac{S_t+q_tV_t}{S_t+q_tV_t+V_t/m_t}
=\frac{\rho_t+m_tq_t}{\rho_t+m_tq_t+1},
\qquad \rho_t=\frac{m_tS_t}{V_t}.
$$

The equal-old/new-covariance-shape approximation yields
$\pi_t^{\rm cov}=m_tq_t/(1+m_tq_t)$ when $S_t=0$. Do not substitute the
stationary-action approximation $c/2$ for tracked $q_t$. A Kalman gain is not
the learner action. The filter supplies movement/uncertainty information to
this risk approximation; it does not establish global optimality.

Initial safeguards are $\pi_{\min}=10^{-4}$ and $\pi_{\max}=.2$, with 16
arrival cold-start actions at the selected fixed control value. Record raw
recommendations, applied values, boundary fractions, and fallback reasons.
The positive floor avoids a zero-action lockout; it is not evidence that the
floor is optimal. A boundary-dominated pilot permits one documented expansion
or contraction within $(0,1)$, applied consistently to adaptive comparisons.
Do not tune safeguards from fresh-replication outcomes without a new revision.

If movement/noise estimates are nonfinite, unsupported, or yield an effectively
zero denominator, use the bounded covariance-only recommendation for that
action and record the reason. The fixed cold start takes precedence while
active. This is an explicit safety fallback, not silent evidence of a working
movement estimator. Report its fraction and never conceal a failed filter
behind the fallback's predictive performance.

### Common-Anchor Observations

Let $a$ be a predictable parameter anchor, fixed while comparing consecutive
observations. Measure **observed-label** scores

$$
\overline s_t(a)=\frac1{m_t}\sum_{(x,y)\in B_t}
\nabla_a\log p_a(y\mid x).
$$

Do not integrate model labels in this observation: that would make its mean
zero and erase the intended discrepancy signal. Model-label integration is
still required for the Fisher metric. This distinction must appear in both
code comments and the notebook glossary.

For an approximately correct local model,
$\mathbb E[\overline s_t(a)]\approx\mathcal I(a)(\theta_t^\star-a)$.
The economical observer replaces that map by a diagonal approximation. Define
$A_a=\operatorname{diag}(a_i^{F})$, with $a_i^{F}$ the diagonal Fisher at the
anchor, and let $A_{a,\epsilon}$ use a recorded positive numerical floor only
for the observer solve. Observe

$$y_t=A_{a,\epsilon}^{-1}\overline s_t(a).$$

Then $y_t$ is a noisy proxy for a local parameter discrepancy, not an observed
true parameter or an exact population drift. Off-diagonal Fisher structure,
misspecification, and weak coordinates can invalidate this proxy. Record
floor usage and noise amplification. A common scalar normalization or bounded
coordinate reliability treatment may be repaired in development, but cannot
silently discard most coordinates or claim the diagonal approximation exact.

For measurement noise, maintain the diagonal within-batch sample covariance
of observed scores, divided by $m_t$ and the squared entries of
$A_{a,\epsilon}$. Pool **within-batch centered** variances over an observation
half-life initially 32 arrivals; do not estimate noise by the uncentered
variation of batch means across a drifting population. With $m_t<2$, use a
previously calibrated variance and mark the fallback. Variance estimates used
in an action are lagged. They are sandwich/measurement quantities, not Fisher
observations. Initialization supplies a disjoint-from-holdout calibration of
their scale; never force equality to the Fisher by definition.

The corresponding one-observation local parameter-noise approximation is
$K_i=\operatorname{Var}(s_i)/[a_i^F]^2$ using the same recorded floor. Given the
predictable risk diagonal, use $V_t=\sum_i\widehat d_{t-1,i}K_{t,i}$.
This scale is shared in construction between movement-aware policies and
cancelled algebraically in the covariance-only condition. It is an explicit
diagonal approximation, not an inverse full-Fisher claim.

### Observation-Clock Kalman Filter

Maintain a two-component state per coordinate: local discrepancy $z_i$ and
velocity $v_i$. Time advances once per fixed-size incoming batch, including
stationary holds. The default transition/measurement model is

$$
\begin{bmatrix}z_{t+1,i}\\v_{t+1,i}\end{bmatrix}
=F\begin{bmatrix}z_{t,i}\\v_{t,i}\end{bmatrix}+w_{t,i},
\quad F=\begin{bmatrix}1&1\\0&1\end{bmatrix},
\quad y_{t,i}=H\begin{bmatrix}z_{t,i}\\v_{t,i}\end{bmatrix}+e_{t,i},
\quad H=[1,0].
$$

Only $2\times2$ state covariances per coordinate are needed, not a dense
$974\times974$ covariance. The independence approximation is explicit. Initialize
mean discrepancy/velocity at zero with a finite covariance based on the
initial nominal information mass and measured score variance; record the
resolved values, not an undocumented diffuse prior.

Use $Q_i=\eta K_{0,i}\begin{bmatrix}1/3&1/2\\1/2&1\end{bmatrix}$ as an initial
integrated-white-acceleration process model. Calibrate one shared dimensionless
$\eta$, initially $\{10^{-4},10^{-3},10^{-2}\}$, rather than hundreds of unrelated
noise parameters. These are pilot proposals, not inferred truths. Freeze the
initial scale $K_0$ for a version; updating measurement-noise estimates does
not automatically redefine process noise. Use standard predict/update equations
and the Joseph covariance update, symmetrizing each $2\times2$ covariance.
Document any positive-definiteness repair and its frequency.

At action time, the filtered state uses observations through the preceding
batch. The next displacement prediction has mean $\mu_v$ and diagonal variance
$P_{vv}+Q_{zz}$. The initial Kalman policy uses its conditional second moment:

$$S_t^{\rm KF}=\sum_i\widehat d_{t-1,i}
\left(\mu_{v,i}^2+P_{vv,i}+Q_{zz,i}\right).$$

This is posterior expected movement energy **under the filter model**, not a
proven unbiased frequentist estimate of realized population movement. Plot its
mean-motion and uncertainty contributions separately. Nonzero uncertainty is
not evidence that the environment moved. Do not subtract a posterior covariance
and call the result automatically debiased. If process uncertainty alone drives
aggressive actions through a quiet hold, diagnose/calibrate it before allocating
large runs. A mean-only plug-in ablation is a separately named bounded fallback,
not a hidden replacement of the stated recommendation.

### Anchor Refresh Without Invented Movement

Start with the initialization anchor. At a predictable cadence initially 64
arrivals, refresh to the accepted learner parameters, never to an oracle.
For a change $a\to a'$, recenter $z\to z+a-a'$ and leave velocity unchanged.
This preserves the implied physical location $a+z$ under the local coordinate
model. It does not guarantee validity when the diagonal score map changes.

Audit that approximation by evaluating the latest available training batch at
both anchors, using its observed labels and matching normalization. Record
the discrepancy after recentering and compare it with measurement uncertainty.
Do not assimilate these repeated examples as an independent extra batch or
advance $q_t$. Refresh the anchor Fisher diagonal from explicitly budgeted
available inputs; no evaluation inputs or uncharged oracle refreshes are allowed.
Use an already compatible measurement where possible.

Refresh measurement-noise normalization and its raw score-variance state when
the anchor changes. Re-evaluate available paired inputs or explicitly reset
the raw moments; never carry raw variances from one score anchor into another
unchanged. The physical discrepancy covariance is invariant under an exact
translation, but extra mapping error must be diagnosed rather than silently
treated as zero. Do not count paired refresh evaluations as additional evidence.

If frequent false innovations or anchor-change spikes occur, Sol may choose a
longer refresh interval or a bounded reconstruction from at most 32 stored
training examples, with explicit memory accounting. Reconstruct rather than
double-count the old evidence, and version the observer policy. A fixed-anchor
development ablation helps distinguish mapping defects from filtering defects.
Do not add a moving-parameter score stream and interpret its changes as pure
population movement. The need for this calibration is a known risk, not grounds
to hide an unfavorable result.

## Conditions

| Condition | Action policy | Interpretation |
| --- | --- | --- |
| **Fixed control** | Development-selected $c$, constant online | Practical reference with the same diagonal EWC |
| **Covariance only** | $m_tq_t/(1+m_tq_t)$ with common safeguards | Does movement estimation add anything? |
| **Anchor EMA** | Common-anchor level/trend smoothing, then the risk formula | Simpler observation-clock observer control |
| **Anchor Kalman** | State/uncertainty prediction above, then the risk formula | Main treatment |
| Frozen initialization | Holdout inference without training | Context only, not the primary learning comparator |

For Anchor EMA, use a specified Holt-style level/trend recursion on the same
normalized observations, with $\alpha=1-2^{-1/H_{\rm EMA}}$:

$$
\ell_t=(1-\alpha)(\ell_{t-1}+v_{t-1})+\alpha y_t,\qquad
v_t=(1-\alpha)v_{t-1}+\alpha(\ell_t-\ell_{t-1}).
$$

Use $S_t^{\rm EMA}=\sum_i\widehat d_{t-1,i}v_{t-1,i}^2$ at decision time.
This is a point plug-in and may have squared-noise bias; report that explicitly.
It is not the historical angular-clock observer. The Kalman/EMA contrast tests
the complete estimator package, including uncertainty treatment, not solely
the gain formula. Both use the same anchor-management, noise-scale, timing, and
safeguard rules, but their on-policy anchors and actual score observations differ.
A shared-state shadow pilot below provides a cleaner estimator comparison.

Screen $H_{\rm EMA}\in\{8,16,32\}$ and the three proposed process-noise scales
using a shared development driver trajectory. A two-model quiet/moving IMM [2]
is an optional development candidate if a single filter cannot handle both
long holds and transitions: infer regime weights from innovations, never from
known schedule labels. Its introduction, transition probabilities, component
models, and moment mixing must be versioned and tested. Do not automatically
carry every filter variant into fresh replication.

## Pilot Choices And Applied Metrics

Reuse Plan 16's proposed fixed-control screen $c\in\{.01,.025,.05\}$ with a
common ridge starting at $r=.1$. This is a development tuning allowance for a
strong control, not an assumption that $.025$ wins. Make any bounded ridge
repair common across all policies. Do not silently relax EWC only for the
Kalman arm or give it extra online labels.

Use Plan 16's transparent equally weighted acquisition/retention development
ranking as the unattended default for the fixed setting and on-policy
finalists, with measured cost breaking exact ties. The number is only a
candidate-selection heuristic, not a success definition or reported replacement
for the two accuracies. Sol may make a recorded trade-off judgment before fresh
replication. For shadow screening, rank by regime-balanced one-step predictive
innovation log score, with calibration failures and mapping errors shown; pass
at most two candidates per family to the bounded on-policy comparison. A valid
executable default must permit progress without an agent or user selecting a
candidate interactively.

Run observers in shadow on one or two common fixed-policy development paths
first. They may process already available training observations at common
anchors; their recommendations do not alter those paths. Compare stationary
false movement, delayed response, predictive innovations, anchor sensitivity,
and uncertainty calibration. Optional high-sample score means at a few states
are audit-only and budgeted; they cannot choose live actions. On digit-9
mixtures, expected fixed-anchor scores are affine in mixture prevalence with
fixed within-class sampling, which helps audit the observation model. This
does not reveal the population parameter path or certify the diagonal inverse.

Use those observations and short/full-horizon on-policy pilots to choose one
EMA and one Kalman-family policy for fresh replication. Prefer stable candidates
with interpretable acquisition/retention curves; no numerical lift threshold
is a gate. Retain unsuccessful pilot results. Allow one half/double or adjacent
decade expansion when a concrete noise/mapping diagnosis warrants it, not a
large factorial search. Protect time for actual closed-loop learning: good
shadow predictions are not evidence of useful control.

The headline endpoints are the agreed digit-9 recall and macro digit-0--8
classification accuracy:

$$A_9=\Pr(\widehat Y=9\mid Y=9),\qquad
A_{0:8}=\frac19\sum_{k=0}^{8}\Pr(\widehat Y=k\mid Y=k).$$

Also show 9 precision at fixed evaluation prevalence $.1$, false positives,
and expected/realized 9 exposure. Follow Plan 16's independent stratified
holdouts, argmax rule, undefined precision convention, initial/online exposure
accounting, and evaluation cadence. Current versus historical mixture-weighted
accuracy is secondary because it can change solely through prevalence. These
are class-specific acquisition/retention proxies, not different visual tasks.

Report whole-trajectory and regime-specific observation-averaged accuracies,
endpoints, paired percentage-point differences, and measured compute. Show
raw/applied $\pi$, exact weight concentration, realized replacement mass
$\prod_j(1-\pi_j)$, and boundary fractions on a compact aligned panel. The
concentration and nominal replacement mass describe the algorithm's weights;
they are not direct measurements of knowledge or proof of a memory mechanism.

Use nominal 95% intervals from whole paired replicas, with all four primary
policies compared against fixed and the movement policies against covariance
only. Preserve covariance by resampling each entire paired block, not individual
updates or holdout images. Analysis is exploratory with multiple comparisons
and repeated inspection disclosed. Do not hide acquisition/retention trade-offs
inside a newly optimized scalar score or claim that an interval crossing zero
proves preservation. NLL and estimator health are secondary diagnostics.

## Execution Phases

### Phase 0: Sol Implementation And Acceptance

Implement the policies, shared digit-9 learner, frozen-action timing, diagonal
observer model, noise/anchor handling, budgets, and report builder, proposed
under `mnist_experiment/kalman_pi/`. Create `plan17_results.ipynb` immediately
with a near-top progress panel, short glossary, and explicit pending sections.
Reuse compatible Plan 16 helpers without making execution depend on its results.

Tests cover exact $q$ recursion, stationary fixed-action limits, distinction
between model-label Fisher and observed-label scores, scalar/multicoordinate
Kalman parity, Joseph covariance PSD, level/trend recursion, posterior movement
moment, reanchoring invariance, no duplicated evidence, nonzero observation-
clock updates on holds, and no access to current-batch/future data or schedule
metadata when choosing an action. Test projection/observer floors separately
from statistical ridge and Fisher state. Run focused unit tests and tiny
CPU/CUDA interruption/resume integrations, including reporting and budget limits.

### Phase 1: Common Calibration And Resource Pilot

Build development initial assets, time a full fixed trajectory, and collect
shared-state shadow measurements. Audit score/noise scaling and anchor refresh.
Time one on-policy trajectory of each retained policy before forecasting the
full cohort. Display initial predictive curves immediately. Measure controller,
score evaluation, training, holdout inference, I/O, and report costs separately.
Do not charge unused observer work to the fixed control merely to equalize time.
Replace provisional timing/storage forecasts with measured ones.

### Phase 2: Bounded Observer Development

Complete the fixed-control and observer screens with at least two development
replicas where affordable, including stationary and transition intervals.
Choose one candidate per primary adaptive family and publish the reasons and
uncertainty. Compare on-policy evidence, not just shadow agreement. An IMM or
mean-only-energy variant is a bounded replacement candidate, not an obligatory
fifth arm. Numerical/identifiability problems receive a documented repair;
weak efficacy alone does not cause early abandonment or an automatic dataset
switch. Freeze the next paired cohort allocation using measured cost.

### Phase 3: Fresh Paired Exploration

Target **eight new main-schedule paired replicas across four policies** and
**four new stationary paired replicas across them**. Run interleaved paired
blocks so both regimes have visible progress early. Complete all arms of a
block before counting it as a paired result; preserve completed unpaired work.

These targets are adjustable before the cohort starts to respect the 20-hour
cap, with 25% runtime contingency and reserved closeout resources. Prefer fewer
complete pairs over shortening holds, dropping the fixed/covariance controls,
or repeatedly tuning the favored policy. Do not pool development or materially
changed protocols into this fresh cohort. Do not extend solely for significance.

### Phase 4: Applied Interpretation And Closeout

Compare acquisition, retention, precision, and overhead. Describe which regimes
benefit or suffer and whether the fixed or covariance-only policy is equally
good or better. Distinguish better observation design from the full Kalman
estimator package, and distinguish estimator calibration from predictive value.
Plot all important effects even when they conflict with the motivating story.
Finish with a short table and candid exploratory conclusions, not a pass/fail
gate. Leave the handoff and artifact index current; do not launch another plan.

| Phase | Operator | Provisional active compute | Deliverable |
| --- | --- | ---: | --- |
| 0: acceptance | Sol | 0.5--1.5 h | Tested policies, recovery, initial notebook |
| 1: calibration/timing | Sol | 1.5--2.5 h | Shared-state observations and measured forecast |
| 2: development | Sol | 2--3.5 h | Selected observers, visible trade-offs/repairs |
| 3: fresh pairs | Sol | 7--10.5 h | Four-policy applied comparisons |
| 4: closeout | Sol | 0.5--1 h | Paired summaries and concise conclusions |
| **Planning range** | **Sol** | **11.5--19 h** | Subject to full-trajectory timing |
| **Hard allocation** | **Sol** | **20 h** | Includes retries and all computational overhead |

Initial engineering is estimated at 4--8 hours separately, less if compatible
Plan 16 helpers already exist. Computational tests count toward the active
allocation. These are planning estimates, not measured runtime guarantees.
Protect at least 0.5 hour for closeout; omit unstarted optional work first.
If necessary, finish with fewer pairs and an explicit resource limitation.
Additional compute requires a new user allocation rather than automatic scope
growth. A reboot or new invocation never resets the cumulative counter.

## Human-Friendly Progressive Report

The canonical report is **`mnist_experiment/plan17_results.ipynb`**, directly
beside this plan. Use roughly six multi-panel figures and three compact tables:

1. Near-top live progress: current phase/revision, complete/planned paired blocks
   and trajectories by regime, active unit, checkpoint, heartbeat, active hours,
   allocation, forecast remaining, and storage headroom. Label optional work.
2. Brief motivation/settings/glossary: action versus prevalence, score anchor,
   observation clock, movement versus measurement noise, and the two accuracies.
3. Chronological development results: one calibration panel, one compact table
   of settings and significant changes, and acquisition/retention trade-offs.
4. Accuracy/precision trajectories with regime bands, uncertainty, frozen-model
   reference, and expected/realized 9 observations prominently labeled.
5. Aligned actions, movement/uncertainty contributions, and nominal memory-weight
   history; paired accuracy-effect and actual compute tables/plots.
6. Bottom Conclusion with useful findings, limitations, and remaining questions.

Keep older phases before newer ones, use matched axes where sensible, include
every reference in legends, and state favorable signs. No giant configuration
dump, per-replica figure wall, or chronological debug transcript. Link full
diagnostics from compact artifact tables. Scientific repairs appear promptly
in a small significant-changes table, not only in the final conclusion.

The notebook does no training, inference, downloads, Fisher estimation, repairs,
or automatic resume. Python entry points produce compact sealed summaries and
bounded bootstrap results. Explicit pending/partial status is valid; corrupt or
incompatible completed artifacts are not silently skipped. Refresh atomically
at stage boundaries/safe stops and at most every 15 active minutes. Resume
must not replay all earlier report builds.

## Resumption And Autonomous Operation

Inherit Plan 16's immutable runs, stop-aware validation/reporting, checkpoint
and source-identity tests, venv requirement, single-writer lease, cumulative
budget accounting, and shared **500,000,000,000-byte peak storage ceiling**.
Target under 10 GB of new Plan 17 artifacts, revised from the pilot. Count
shared assets once, staged/checkpoint copies, and finalization reserves; keep
at least 20 GB actual filesystem headroom. Do not prune historical evidence,
interfere with archival jobs, or migrate data without authorization.

Use `cache/mnist_experiment/plan17/<study-id>/` with named seeds for initialization,
stream, reference, evaluations, schedule variation, and bootstrap. Save actions,
nominal weight concentration, Fisher vector, parameters, accepted solver state,
both observer components/covariances, measurement-noise moments, anchor/Fisher
scales, refresh counters, optional reconstruction IDs, pending batch, RNG states,
metric offsets, and active time/storage counters atomically at accepted arrivals.
Resume must neither repeat assimilation nor count repeated examples as new.

Require `--resume`, `--allocation-hours`, `--max-wall-seconds`, `--max-units`,
`--clear-stop`, status and stop commands, and signal/stop-file handling. Check
for stop requests through training, reference generation, validation, analysis,
and reporting. Target graceful exit within one minute and loss of at most one
arrival or bounded initialization/analysis chunk on abrupt interruption. Test
the actual bounds; disclose unavoidable kernels. An incremental validated index
must avoid full historical replay on each resume.

The scheduler advances all authorized phases without agent polling. It may run
inside `tmux` when remote operation requires persistence, but local operation
does not require it. A reboot still requires an explicit resume. Sol may
autonomously correct numerical, mathematical, or engineering defects and make
bounded evidence-driven revisions without returning to Astra. Record the
evidence, before/after equations, validity of prior units, tests, and updated
forecast; changed scientific behavior receives a new source/config identity.
Never rewrite completed artifacts or pool incompatible revisions. Preserve
useful unaffected work and report limitations rather than keep the node idle
awaiting a routine model switch. This autonomy does not override resource caps.

Implemented interfaces (active calibrated revision; original configurations
remain available for reproducing earlier development probes):

```bash
~/.venv/bin/python -u -m mnist_experiment.kalman_pi.run \
  --config mnist_experiment/kalman_pi/configs/calibrated.json \
  --root cache/mnist_experiment/plan17/production_v4 \
  --allocation-hours 20 --resume
~/.venv/bin/python -m mnist_experiment.kalman_pi.control status --root <study-root>
~/.venv/bin/python -m mnist_experiment.kalman_pi.control stop --root <study-root>
~/.venv/bin/python -m mnist_experiment.kalman_pi.report --root <study-root>
```

Sol records verified launch/resume commands in
`mnist_experiment/PLAN17_HANDOFF.md` once implementation begins. Smoke and
production use distinct protocol identities. Neither plan authorization nor
report generation authorizes resuming another historical experiment.

## References And Evidence

1. Y. Ollivier, [Online Natural Gradient as a Kalman Filter](https://arxiv.org/abs/1703.00209), 2018 revision. The equivalence has specific assumptions; it does not identify our filter gain with the EWC action.
2. H. Blom and Y. Bar-Shalom, [The Interacting Multiple Model Algorithm for Systems with Markovian Switching Coefficients](https://doi.org/10.1109/9.1299), 1988. Background for the optional quiet/moving observer.
3. S. Sarkka and L. Svensson, [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/), 2023. Standard state-space filtering and covariance updates, not a guarantee that our observation model is calibrated.
4. [Plan 7](plan7.md), Phase 5. Historical movement-energy inflation and covariance-only calibration motivate the controls; they are not fresh observations in this study.
5. [Mathematical overview](../mathematical_overview.ipynb), fixed-batch risk and statistical-realization appendices. Keep the nominal weight recursion, population assumptions, and causal policy distinct.

## Execution Closeout

Phases 0--4 are complete. The final scientific unit sealed at
2026-11-02 04:19 UTC, followed by artifact/report validation. The full planned
fresh cohort was retained; no additional study or fallback dataset was started.

Development made three documented revisions before fresh outcomes existed:

- Frequent anchor refresh caused severe noise-normalization collapse. A
  fixed-anchor ablation removed the extreme innovation spikes.
- The original 256-example noise-calibration subset contained no 9s. Calibration
  now uses all 2,048 already-seen initialization examples, adding no labels.
- One adjacent-decade process-noise screen retained
  $\eta\in\{10^{-5},10^{-4},10^{-3}\}$. Earlier roots and their results remain
  immutable development evidence and are not pooled into fresh inference.

Selection froze at 2026-11-01 22:04 UTC: $c=.025$, ridge ratio $.1$, EMA
half-life 16 incoming batches, and Kalman $\eta=10^{-4}$. All policies use
direct-EMA diagonal Fisher estimates; no LFU, STM, replay training, or FD KL
was introduced. The realized study contains 22 ordinary development paths,
two separate shadow drivers, and 48 fresh paths. An actual node restart was
recovered through the accepted-arrival checkpoint without changing these choices.

**Applied findings:** on changing schedules, Kalman versus fixed changes
observation-averaged 9 recall by **-0.94 pp [-1.63, -0.27]** and 0--8 retention
by **+1.03 pp [0.57, 1.48]**, with **3.4%** more measured whole-unit wall time.
Stationary differences are **-1.25 pp [-2.22, -0.34]** and
**+0.94 pp [0.48, 1.61]**, respectively. This is a repeatable trade-off, not
dominance over the tuned fixed policy.

Against covariance-only on changing schedules, Kalman improves both outcomes:
**+4.17 pp [3.42, 5.12]** 9 recall and **+1.38 pp [0.40, 2.35]** old retention.
EMA adds only very small effects over covariance-only. The complete Kalman
package has applied value relative to that control, but this does not isolate
genuine-motion detection from its uncertainty contribution.

Regime-specific effects are not uniform. During the return ramp, Kalman versus
fixed gains **4.44 pp [1.69, 7.08]** recall and **1.65 pp [0.60, 2.65]** retention;
the final low hold instead loses **3.95 pp [-5.08, -2.92]** recall. All five
predefined regimes are shown in the notebook. Every interval here is nominal
95%, from whole-paired-replica resampling, exploratory under multiple comparisons
and interim inspection. Equal exposure does not establish sample savings.

**Resources and verification:** approximately 4.14 charged active hours out of
20; approximately 1.6 GB of shared accounted data out of the 500 GB ceiling.
The focused suite passes 173 tests; all 102 units passed full-file verification,
all 12 fresh blocks share their required pairing assets, and all six embedded
figures passed nonblank checks. Completed-study `--resume` is a no-op. Frequent
recovery-checkpoint I/O is substantial here, so wall-time ratios are not
deployment FLOP estimates. The fixed-anchor result does not establish a robust
ongoing reanchoring mechanism.

Canonical report: [plan17_results.ipynb](plan17_results.ipynb). Under the completed
root, `selection.json` records the frozen cohort, `report/latest.json` indexes
the final scalar analysis/figures, `closeout_validation_final.json` records final
integrity checks, and `changes.json` preserves the development evidence.

## Phase 5: Acquisition At Comparable Retention

**Motivation:** continual learning with scarce data and compute needs useful
new learning without inadvertently buying it through greater forgetting.
Phases 0--4 calibrated observer noise and selected policies by an equally
weighted acquisition/retention score. They did **not** calibrate the methods
to comparable retention. Common ridge settings did not imply common retention:
Kalman versus fixed gained about 1.03 percentage points (pp) of mean 0--8
accuracy while losing 0.94 pp of mean 9 recall on changing schedules. Those
results describe real operating points, but leave acquisition at comparable
retention unanswered. This follows the lesson of Plan 15 Phase 8.1, without
assuming that calibration will rescue an adaptive policy.

> At comparable observation-averaged 0--8 retention, does an independently
> calibrated adaptive policy improve digit-9 acquisition over fixed
> $\pi=.025$, and at what measured compute cost?

**Status and scope:** execution authorized on 2026-11-02. Phase 5A is complete:
95 focused tests and CPU/CUDA recovery smokes passed. Phase 5B production
calibration is running in `retention_v1`. Astra specifies this phase;
Sol implements, calibrates, runs, and closes it out
autonomously under the bounded rules below. Preserve every historical run and
the original notebook results. Append this phase after the original conclusion.
The resource ceiling remains **20 cumulative active hours for Plan 17**, not
20 additional hours: about 4.14 hours have been spent, leaving about 15.86
before this follow-up's tests, reporting, and runs. Read the live ledger before
execution. This phase does not authorize another plan or dataset.

### Fixed Reference And Calibration Variable

Keep the reference at $\pi_t=.025$ and ridge ratio $r=.1$. Rerun it on each
new paired development/confirmation stream; historical control outcomes are
context, not the new comparison sample. Retain all three adaptive families:
covariance-only, anchor EMA with half-life 16 incoming batches, and anchor
Kalman with $\eta=10^{-4}$. Keep the completed v4 observer/noise settings,
fixed initialization anchor, 2,048-example initialization noise calibration,
action bounds, cold start, and solver settings. Do not reopen observer tuning.

The only efficacy calibration variable is a nonnegative ridge ratio $r$,
constant within a trajectory. Its existing physical objective is

$$
J_t(\vartheta;r)=\overline L(B_t;\vartheta)
+\frac{1-\pi_t}{2\pi_t}
\sum_{i=1}^{d}(\widehat d_{t,i}+\kappa_t)
(\vartheta_i-\theta_{t,i})^2,
\qquad
\kappa_t=r\,\frac1d\sum_{i=1}^{d}\widehat d_{t,i},\qquad d=487.
$$

Thus ridge augments the EWC curvature inside the same old/new odds factor.
It does not change the stored Fisher diagonal or create a second Fisher
estimand. At $r=0$ only statistical ridge vanishes; diagonal EWC and the
existing numerical coordinate-scaling floor remain. Freeze all objective
quantities within each solve. One predictable action still drives both EWC
and direct-EMA Fisher updating, and its actual values drive the exact $q$
recursion. No post-solve update scaling or online retention controller is added.
Changing ridge can change subsequent states/actions; do not replay actions
from another ridge path as though they were the new policy's own decisions.

For this phase only, independently selected ridge ratios replace the original
requirement for common ridge across policies. Give all adaptive families the
same search allowance. The estimand is the performance of independently
calibrated policy packages against the unchanged reference, not an isolated
causal effect of the observer or a proof that its risk formula is optimal.
No new multiplier on EWC odds or remapping of $\pi$ is an automatic fallback.

### Retention Target And Selection

Let $j$ index paired replicas, $m$ policies, and $s$ the changing or stationary
schedule condition. At recorded evaluation checkpoints $n_{j,k}$, let
$a^9_{m,j,k}(r)$ be held-out digit-9 recall and $a^O_{m,j,k}(r)$ the unweighted
mean of held-out class accuracies for digits 0--8. Higher is better. Define
the existing observation-averaged statistic explicitly, for $D\in\{9,O\}$:

$$
\overline A^D_{m,j}(r)=\frac{1}{N_j}
\sum_{k=1}^{K_j}(n_{j,k}-n_{j,k-1})
\frac{a^D_{m,j,k-1}(r)+a^D_{m,j,k}(r)}{2},
\qquad n_{j,0}=0,\quad n_{j,K_j}=N_j.
$$

This is trapezoidal averaging on the actual incoming-digit clock, including
the shared initial checkpoint. It estimates performance throughout the run,
not only at its endpoint. Each full replica has equal weight even when its
predrawn schedule is longer. Evaluation observations never train the model,
update an observer/Fisher, or enter a live action. Predictive NLL contains
no regularization penalties and remains secondary.

On development replicas for schedule $s$, calculate paired differences in pp:

$$
\widehat\Delta^O_{m,s}(r)=\frac{100}{J_s}
\sum_{j=1}^{J_s}
\left[\overline A^O_{m,j}(r)-\overline A^O_{\rm fixed,j}(.1)\right],
\qquad
\widehat\Delta^9_{m,s}(r)=\frac{100}{J_s}
\sum_{j=1}^{J_s}
\left[\overline A^9_{m,j}(r)-\overline A^9_{\rm fixed,j}(.1)\right].
$$

The prospective **development matching band is $[-0.5,+0.5]$ pp** for
$\widehat\Delta^O$. This is an illustrative calibration resolution, smaller
than the historical approximately 1 pp Kalman retention surplus; it is not
an application-derived safety limit, an equivalence claim, or a significance
target. Keep it fixed before running the grid. Publish development uncertainty
and the actual signed gap; do not widen the band to admit a favored method.

For each method and schedule, among measured finite candidates inside the
band select the highest mean 9 recall. Break exact ties by smaller absolute
retention gap, then lower measured learning time, then smaller ridge ratio.
If no candidate enters the band after bounded refinement, select the closest
retention candidate, then break ties by higher 9 recall, lower learning time,
and smaller ridge, and label it
**unmatched**. Carry it into confirmation as an informative trade-off; do not
call its acquisition contrast retention-matched. Numerical failures are
recorded explicitly and never converted to completed finite candidates.
If an entire policy has no numerically valid candidate after a documented
repair attempt, report it as non-estimable; do not silently substitute another
policy. Development cohorts are small, especially stationary, so even a close
development match can miss on confirmation.

Calibrate changing and stationary conditions separately; never average them
into a selection score. Each resulting ridge is frozen for the entire
trajectory and schedule condition, not switched between its regimes. This
tests policies configured for two known experimental conditions, not one
universal policy that detects which schedule it is in. Schedule labels and
future boundaries remain unavailable to online controllers.

### Phase 5A: Implementation And Recovery Acceptance

Use a separate protocol/root, for example
`cache/mnist_experiment/plan17/retention_v1/`. Add the smallest experiment-local
configuration/scheduler/report extension needed; retain the original v4
execution path and source manifest. A Phase 5 report must retain the original
report section and append new results. Once Phase 5 results exist, prevent a
historical report-only invocation from silently replacing the combined
canonical notebook. Keep old report generations and scalar artifacts intact.

Use separately named development/confirmation seeds and the existing isolated
data-role scheme. Development evaluation panels may select ridge; confirmation
panels may not. Initialization, online, reference, and evaluation IDs retain
their disjoint roles. All candidates within a replica share initialization,
initial Fisher, exact observations, schedule, and evaluation panel. Historical
confirmation results motivate this design but must not enter the new ridge
selection. New seeds provide fresh trajectories, not a new external dataset;
disclose repeated research on MNIST.

The original analyzer groups comparisons by common ridge. **Do not use that
grouping for Phase 5.** Store a frozen policy-to-ridge map for each schedule and
compare the explicitly selected treatment against the fixed $r=.1$ arm using
the same replica/assets/stream identity. Validate both expected ridge values;
do not merely drop ridge from global pairing checks. Keep the historical
analysis contract intact. Tests must reject unintended candidate pairing,
changed selection on resume, and development/confirmation pooling.

Also test weighted accuracy means, pp units, band boundaries, tie-breaks,
unmatched selection, bounded refinement, no confirmation leakage, persistent
budget accounting, and idempotent candidate scheduling. Run tiny CPU/CUDA
recovery smokes including abrupt interruption of a calibrated-policy path.
Preserve model/Fisher, observers, action history, RNGs, selection decisions,
metric offsets and resource counters; accepted arrivals must not be repeated.
Continue supporting `--resume`, status/stop, cumulative and invocation limits,
and stop checks through validation/reporting. Publish verified commands in
`PLAN17_HANDOFF.md`; the existing completed v4 command is not a Phase 5 launch.

### Phase 5B: Bounded Retention Calibration

Use **four new changing-schedule and two new stationary development replicas**.
Preassign two changing replicas to each length factor in $\{1,1.25\}$ and one
stationary replica to each, so selection sees both intended horizon lengths.
The factors change every hold/ramp consistently and are fixed before outcomes;
do not inherit the old helper's forced factor 1 for all development paths.
Pair every candidate within each replica and use complete trajectories.

Screen each adaptive policy over

$$r\in\{0,.01,.03,.1,.3,1\}.$$

Including zero identifies whether removing ridge can reach the reference;
the existing $r=.1$ setting gives a same-cohort link to the original design.
The unchanged fixed arm runs once per replica, not once per candidate. The
initial grid is **114 trajectories**: $6(1+3\times6)$. Reuse sealed compatible
initialization/noise assets across candidates, never across incompatible seeds.

After each completed candidate block, show acquisition versus retention,
method colors, actual ridge labels, the fixed reference and matching band,
with changing/stationary panels separate. Show every measured setting and
development uncertainty. Connections guide the eye; they do not imply a
monotone ridge response or an attainable interpolated policy. Inspect numerical
health, especially the zero-ridge boundary, before interpreting the frontier.

For a method/schedule with no measured setting in the band, allow at most
**two additional ridge values**, each evaluated on all its development replicas:

- If neighboring tested ridge values bracket a zero retention gap, choose the
  bracket with smallest sum of absolute endpoint gaps and test its geometric
  midpoint (arithmetic midpoint when one endpoint is zero). Break bracket ties
  by smaller lower ridge. Update the bracket from measured results only.
- Without a bracket, refine the interval incident to the closest-gap candidate
  whose other endpoint has the next-smallest absolute gap; use the same midpoint
  rule. If the closest candidate is instead the upper boundary and its two
  highest tested settings show decreasing absolute gap with the same sign,
  one next value may be three times that boundary, at most $r=3$.
- Never search negative ridge or infer that a zero-ridge miss can be fixed by
  weakening other regularizers. Stop refinement upon entering the band or
  exhausting the two-value allowance; finite unmatched outcomes are publishable.

This adds at most **36 trajectories** across all methods/schedules. A new
candidate always requires actual learning; interpolated predictions or model
parameters cannot stand in for a run. Save decisions before launching their
units so interruptions neither duplicate nor alter the search. Do not stop
development simply because acquisition looks unfavorable. At most 150
development trajectories precede confirmation; avoid a new factorial search.

Use pilot timing to reserve the full confirmation and closeout allowance
before optional refinement. Freeze the method/ridge map, matching status,
cohort sizes, seeds, horizons, and analysis definitions before any confirmation
outcome is exposed. No p-value or confirmation retention gap selects settings.

### Phase 5C: Fresh Paired Confirmation

Target **eight new changing and four new stationary replicas across all four
policies**, or **48 trajectories**. Balance length factors within each schedule
(four/two replicas per factor respectively). Rerun the fixed reference on every
stream. Interleave complete four-arm blocks and both schedule conditions so
progress is interpretable early. Retain every finite completed block, including
unfavorable or retention-unmatched outcomes. Never select confirmation replicas
by realized retention or discard an arm for failing to achieve significance.

Freeze this allocation after measured timing with 25% remaining-run contingency.
If needed, first omit unstarted optional refinement, then reduce fresh cohorts
to balanced complete blocks before confirmation starts. Preserve at least four
changing and two stationary target blocks when feasible; otherwise close with
the available evidence and an explicit resource limit. Do not shorten regimes
or drop a primary policy to manufacture more apparent replication. A hard
interruption can leave partial blocks; show them without treating them as
complete paired evidence. Resume fills them under the same frozen settings.

Report paired whole-replica 95% bootstrap intervals for both accuracy differences
and measured costs, with the same 2,000 draws and persisted named seeds as the
original analysis convention. Use shared block draws across endpoints/contrasts,
separate schedule cohorts, and all three adaptive-versus-fixed comparisons;
movement-versus-covariance comparisons are secondary calibrated-package contrasts.
Intervals are conditional on development-selected settings and exploratory
under multiple comparisons, prior benchmark use, and intermediate inspection.
Do not resample individual checkpoints as independent replicas.

A retention interval crossing zero is compatible with no difference; it is
not evidence of equality. Show its width and the actual point estimate relative
to the development band. A material remaining gap qualifies the acquisition
comparison even when nonsignificant. Do not retune after confirmation, regress
retention out of the accuracy contrast, or add replicas to chase either endpoint's
significance. Matching the whole-path mean does not match every regime or class.

### Phase 5D: Human-Friendly Interpretation

Append a concise section titled **Phase 5: Acquisition At Comparable Retention**
to `plan17_results.ipynb`, beginning with the motivation above and a one-sentence
question. During planning it must say **drafted, not started**. During execution,
update the near-top progress with new-phase development/fresh counts by schedule,
current checkpoint, active hours against the cumulative allocation, forecast and
storage. Preserve old plots, tables, conclusions, and their report generation;
label them as the original common-ridge operating points, without claiming they
are invalid or that this follow-up has already explained them.

Keep the added main story to approximately three figures and two compact tables:

1. Development acquisition/retention map with the band, fixed reference, all
   measured candidates, and selected settings. A settings table lists policy,
   schedule, selected ridge, development gap, and matched/unmatched status.
2. Fresh 9 recall and 0--8 accuracy trajectories in paired columns and separate
   schedule rows, with aligned axes, initial references, and exposure counts.
   Distinguish length cohorts; do not disguise temporal imbalance by pooling.
3. Paired-effect panels for acquisition and retention with 95% intervals and the
   matching band on the retention panel. A final table gives both gaps/intervals,
   complete-pair counts and measured time ratios. Link full regime/class-level
   diagnostics; explicitly discuss contrary regime effects when present.

Conclude whether there is an acquisition gain at plausibly comparable retention,
a remaining trade-off, an uncertain result, or an unattainable calibration target
within the tested ridge family. Describe measured development and online costs
separately; repeated checkpoint I/O remains a tiny-model timing limitation.
This equal-exposure accuracy study does not establish actual samples saved.
Do not introduce a combined accuracy score as a substitute for the two outcomes.

### Phase 5 Resources And Completion

Completed v4 fresh trajectories had median online wall times of about 158--190
seconds across policies. New ridge settings can change solver work, so use these
as an initial forecast rather than a guarantee. Replace the forecast after the
first complete development blocks and measure initialization, observer calibration,
training, evaluation, checkpoint/report I/O, and tests. Charge reused compute
once in the allocation ledger; label counterfactual per-method deployment costs
separately where shared assets would have to be computed independently.

| Sub-phase | Additional active wall-clock estimate | Deliverable |
| --- | ---: | --- |
| 5A: implementation acceptance | 0.25--0.75 h | Tested pairing, selection, resume, initial section |
| 5B: initial grid and initialization | 5.5--6.5 h | 114 full paths and visible trade-off maps |
| 5B: bounded optional refinement | 0--2 h | At most 36 additional paths |
| 5C: fresh paired confirmation | 2.5--3.5 h | Target 48 paths, both accuracies and costs |
| 5D: reporting and closeout | 0.25--0.5 h | Concise interpretation and preserved evidence |
| **Expected additional active compute** | **8.5--13.25 h** | Approximately **9--14 h** for planning |
| **Remaining family ceiling at drafting** | **Approximately 15.86 h** | Reconcile live cumulative ledger first |

Initial programming is provisionally **2--4 hours of engineering**, separate
from computational runtime; the computational tests above are included. Elapsed
time to finish includes both, plus any user pauses. All previous Plan 17 time
and this phase's retries/reporting count toward the 20-hour family cap. The
25% forecast reserve must fit the remaining allowance; the upper grid/refinement
and confirmation forecasts are not promises to exceed it. Reserve at least
0.5 hour for final reporting/validation when allocating remaining runs.

Retain the shared **500 GB** storage ceiling, at least **20 GB** actual free
space, and target **under 10 GB** new Phase 5 artifacts. Preserve immutable
completed units, bounded recovery checkpoints, historical reports, source/config
identities and selection decisions. Stop cleanly for resource limits; do not
prune evidence, migrate data, or reset counters. After execution authorization,
Sol may repair defects autonomously with versioned scientific changes and visible
notes. Enlarging the efficacy search, changing the retention tolerance/target,
or replacing the fixed reference requires a new design discussion, not a hidden
repair. Close out with reproducible commands and findings, including failure to
match retention, without launching another experiment.

### Phase 5A--D Execution Closeout

Completed on 2026-11-02 Pacific time (2026-11-03 00:22 UTC). All 114 initial
grid paths, six stationary refinement paths and all 48 fresh paths completed;
no numerical candidates failed. All six selected operating points were within
the development band, but **none of the fresh retention point gaps remained
within it**. Every adaptive-versus-fixed 9-recall interval was below zero.
These remain acquisition/retention trade-offs, not a successful isolated test
of acquisition at matched fresh retention.

For changing schedules, Kalman minus fixed gives mean 9 recall
$-7.69$ pp (95% interval $[-11.06,-4.91]$) and old accuracy $+1.10$ pp
$[-0.12,2.21]$. Stationary counterparts are $-1.67$ pp $[-3.14,-0.88]$
and $+0.62$ pp $[-0.50,1.80]$. Crossing zero does not establish equal retention.
The full six-row table and regime caveats are in the notebook. No fresh outcome
selected new settings or extended the cohort; Phase 5E was approved from
development evidence before these confirmation outcomes.

The family used approximately **12.76 active hours** at primary completion.
All 205 primary and 102 historical sealed units passed full checksum checks,
with unchanged scientific sources and all six historical embedded figures
preserved. Validation is recorded in `retention_v1/phase5_closeout_validation.json`.
The full regression suite passes 376 tests; final reporting changes pass
66 focused tests. After the primary worker exited, reporting-only changes
protected the forthcoming Phase 5E appendix, reduced candidate-label clutter,
and grouped repeated conclusions. Scalar analyses and scientific units did not
change; earlier report generations remain sealed.

### Phase 5E: Supplemental Zero-Ridge Confirmation

**Approved on 2026-11-02 while primary development was running.** At zero ridge,
the four changing development replicas show improved mean acquisition and
retention for all three adaptive policies relative to the original fixed
reference. These points are far outside the retention-matching band. They are
worth checking independently, but cannot identify an adaptive-policy benefit
because their ridge also differs from the reference. This supplement does not
change Phase 5B selection, the matching tolerance, or Phase 5C confirmation.

Freeze the following three arms before any supplemental confirmation outcomes:

| Arm | Action policy | Constant ridge ratio $r$ |
| --- | --- | ---: |
| Original reference | Fixed $\pi=.025$ | .1 |
| Zero-ridge control | Fixed $\pi=.025$ | 0 |
| Zero-ridge Kalman | Kalman, $\eta=.0001$ | 0 |

Retain all other learner, diagonal Fisher, observer, initialization and numerical
settings from Phase 5. Zero ridge removes only the extra isotropic term, not
diagonal EWC. No LFU, STM, replay training or new hyperparameter search is added.
Run **eight new changing-schedule replicas**, three paired arms each, for
**24 trajectories**. Predraw four replicas per existing horizon factor
$\{1,1.25\}$. Preserve the primary study-wide data-role and development/fresh
partitions; use a separate named seed namespace for supplemental initialization,
streams and holdout sampling. Within each replica share exact initialization,
initial Fisher, stream and holdouts. These are independent new replica draws,
not recycled development trajectories or additional primary-confirmation pairs.

Use a separately identified computational root and immutable pre-confirmation
design artifact. Complete each three-arm block before advancing to the next
replica. Execute after the primary Phase 5A--D, subject to the same remaining
**20-hour cumulative family allocation**, storage ceiling and closeout reserve.
Estimate **1.5--2 additional active hours**; replace this estimate with measured
costs. Do not reduce the approved primary study to prioritize this supplement,
extend for significance, or silently increase the allocation. If a hard resource
limit intervenes, retain all units and report complete paired blocks and any
partial blocks explicitly. Support `--resume`, accepted-arrival checkpoints,
cooperative stops, source/config identity checks and the existing family lock.

Report both full-path mean 9 recall and old 0--8 accuracy for three paired
contrasts: Kalman/0 minus fixed/.1, fixed/0 minus fixed/.1, and Kalman/0 minus
fixed/0. Use shared whole-replica bootstrap draws (2,000; named persisted seed)
and nominal 95% intervals across both outcomes and contrasts. Retain regime
diagnostics and measured time ratios. No retention matching or selection is
performed in this supplement; any acquisition/retention trade-off remains
visible. This three-cell comparison separates the fixed-policy ridge contrast
from the zero-ridge policy-package contrast, but does not estimate a full
policy-by-ridge interaction, an optimal fixed $\pi$, or actual samples saved.
Its intervals remain exploratory after development-led selection.

Append a concise **Phase 5E: Zero-Ridge Follow-Up** after the primary Phase 5
conclusion in `plan17_results.ipynb`, preserving all previous content and sealed
report generations. Show live progress near the top, a compact paired-effects
figure and table, and short interpretation explaining whether the lead replicates
and whether removing ridge or changing policy accounts for it. Keep fresh
outcomes out of subsequent setting choices.

**Implementation acceptance:** a full combined historical, retention-calibration
and supplemental run passed **376 tests**, including the complete tiny
three-arm CPU scheduler, notebook publication, interruption/resume, independent
source identities, strict paired inference and preservation of inherited outputs.
The nested package leaves the primary scientific manifest unchanged.
Initial 375-test acceptance overlapped the active primary worker and is not
double-charged. The final 376-test rerun and reporting/verification outside
scheduler clocks have a separate conservative 180-second ledger charge.
Final reporting-only changes pass 67 focused tests.
A separately clocked CUDA smoke study completed two three-arm blocks / six
paths across 13 sealed units after an intentional six-unit stop/resume; prior
seals were unchanged. Production started at approximately 2026-11-03 00:32 UTC,
using all eight changing blocks / 24 paths. The family used about 12.81 hours
at supplemental start; no extra allocation was granted.

### Phase 5E Execution Closeout

Completed on 2026-11-02 Pacific time (2026-11-03 02:02 UTC). All **eight fresh
changing blocks / 24 trajectories** completed across three arms, balanced four
per horizon factor. All 49 supplemental units passed full checksum and source
identity checks. The primary data-role partitions are unchanged, while actual
initialization and stream seeds differ from the primary cohort. No outcomes
selected settings or extended the allocation.

Whole-path mean paired differences, with exploratory nominal 95% intervals:

| Contrast | 9 recall gap (pp) | Old 0--8 accuracy gap (pp) |
| --- | ---: | ---: |
| Kalman / zero ridge minus fixed / ridge .1 | +8.64 [3.29, 14.95] | +9.84 [4.00, 16.88] |
| Fixed / zero ridge minus fixed / ridge .1 | +8.61 [4.11, 13.36] | +7.73 [1.81, 14.65] |
| Kalman / zero ridge minus fixed / zero ridge | +0.03 [-2.37, 2.28] | +2.11 [1.56, 2.58] |

**Reading:** removing the extra isotropic ridge already improves both accuracies
under the fixed policy. The acquisition lift against the original reference
therefore cannot be attributed to adaptive $\pi$ alone. At common zero ridge,
Kalman supplies a retention benefit, but a 9-recall benefit is not established.
The latter interval permits losses as well as gains; this is not equivalence
or a claim of cost-free retention. Regime differences remain visible in the
notebook, including some lower 9 recall during quiet holds and the rise.

Against the zero-ridge fixed control, Kalman's mean paired learning-time ratio
is 0.856 [0.828, 0.882], but its online wall-time ratio is **1.037
[1.025, 1.049]**. The table now shows both: excluding evaluation/checkpoint I/O
does not describe total online wall time. These tiny-model timings are not a
large-model FLOP forecast. Shared initialization/calibration is separately
accounted. No actual samples-saved claim is made.

All **356 scientific units** across original, primary and supplemental roots
passed final full-file verification, with unchanged seals and scientific source
identities. Both completed production resume commands returned without training
or changing the notebook. The canonical report retains all inherited cells
and nine figures, and adds two figures; all 11 embedded PNGs are nonblank. The
primary publisher cannot remove the supplement. Detailed validation is in
`retention_supplement_v1/closeout_validation_final.json`; final report generation
is `generation-28161cd6335c51e03e5582cc`.

Final charged family compute is approximately **14.318 hours / 20 hours**,
including historical work, acceptance, CUDA smoke, the primary study,
supplement and completed-resume checks. Workers have exited. No further
experiment, retuning or significance-driven extension is queued.

## Phase 6: Adaptive Weights Under Continuous Registration Drift

**Status: Phase 6A--6E completed on 2026-11-04, including post-reboot integrity,
CPU/CUDA recovery acceptance, 16 development paths and all 48 fresh paths.
Approximately 4.30 of the separately authorized 20 active hours were used.** Astra supplies this design. After execution authorization, Sol
implements, calibrates, executes and reports without requiring model switches
or intermediate user reviews. The proposed resource allowance is specified
below; the completed digit-9 studies remain historical evidence.
The earlier categorical task, four-policy specification and resource tables
describe Phases 0--5. This section explicitly specifies the regression task,
selection, optional ridge companions and proposed resources for Phase 6.

### Motivation And Research Question

The objective remains statistically and computationally efficient continual
learning from small incoming batches. Call $\pi_t$ the **adaptation weight**:
it weights new likelihood information and replaces part of the maintained
Fisher summary. Larger values can improve adaptation while accelerating loss
of historical information. The question is whether varying that weight helps
more than an already useful constant, **$\pi=.025$**.

The digit-9 experiments included long stable periods, smooth increases,
smooth decreases and stationary controls. Their limitation was not an absence
of regimes. Phase 5E found a retention benefit for Kalman at common zero ridge,
but no established acquisition benefit. Removing ridge reproduced the apparent
acquisition improvement under fixed pi. Recorded supplemental checkpoints also
showed little of the intended rise-in-change/fall-in-stability action pattern;
roughly 78--82% of the movement-energy term came from filter uncertainty.
These are diagnoses of the tested implementation, not proof of a universal
Fisher-information obstruction or failure of the local risk minimization.

Small, gradual changes in rare-class prevalence may be difficult to identify
soon enough to act on them. A calculation for the original base schedule gives
KL divergence about .0129 between the first 16 rising batches and continued
low-prevalence sampling. Under equal prior probabilities, Pinsker's inequality
bounds data-based discrimination accuracy by about 54% for those 64 images.
This concerns an early detection delay; it does not show that the entire ramp
is undetectable. A smooth sampling law is necessary for our local interpretation
but does not itself provide an observable change signal.
For reproducibility, this is
$4\sum_{j=0}^{15}D_{\rm KL}(\operatorname{Ber}(p_j)\Vert
\operatorname{Ber}(.005))$, with
$p_j=.005+.095[1-\cos(\pi j/127)]/2$. The equal-prior correctness bound is
$[1+\sqrt{D_{\rm KL}/2}]/2$; see the testing inequalities in
[Duchi's lecture notes](https://stanford.edu/class/stats311/lecture-notes.pdf).
Both hypotheses share the preceding history, and the decision rule is not
told which regime began.

[Plan 15](plan15.md), especially Phases 7 and 10, motivates a different context.
Rotated classification and single-digit landmark regression supplied useful
predictive gains for FD regularization. Phase 10's six-coordinate registration
had a descriptive 14.79% sample-saving estimate, with the documented inverse-
uncertainty qualification. Those findings support a learnable task and useful
metrics; they do not establish adaptive-pi efficacy. Importantly, Phase 10's
temporal shift was query blur under a fixed affine law. This phase instead
changes the **rotation distribution itself**, affecting ordinary incoming
observations rather than a rare novel component.

> **Question:** when continuous visual change is observable early enough to
> matter, can a causal, general-purpose adaptation-weight policy improve
> registration learning and retention relative to fixed $\pi=.025$?

Use one reference/query pair of one digit per observation. Reference images
reduce the ambiguity of recovering absolute orientation from handwriting.
Three virtual landmarks give an applied error in pixels. Rotation-only drift
has one underlying changing degree of freedom even with six output coordinates;
do not claim six independent sources of change. Random nuisance affine factors
remain present but have fixed distributions. Digit grids, scalar companion
tasks, extra landmarks, replay, STM, FD KL and LFUs are outside this phase.

The registration rule itself remains fixed. A model that already generalizes
across all later rotations may have little useful adaptation to perform.
Readiness must therefore assess both observable change and a recoverable
prediction deficit. A negative result with either missing is less informative
about adaptive weighting than a negative result with both present.
Gaussian regression also removes the categorical probability factor from
per-input Fisher curvature, while retaining possible Jacobian ill-conditioning.
It does not remove information limits. In the ideal efficient-MLE Fisher-risk
calculation, $\operatorname{tr}(\mathcal I\mathcal I^{-1})=d$ on an identifiable
regular parameter space; a weak spectrum alone is not a proof that the measured
Kalman uncertainty is irreducible. This study distinguishes predictive value
from the calibration and local-approximation assumptions behind that argument.

### Task, Stream And Data Roles

Reuse Phase 10's tested renderer conventions through versioned dependencies,
without editing its scientific package or reusing its trained checkpoints:

- Input: two $28\times28$ grayscale views of one MNIST source, with an
  antialiased $18\times18$ digit centered on the reference canvas.
- Forward affine map $T(z)=Az+t$, with
  $A=R(\alpha)\operatorname{diag}(e^{a_x},e^{a_y})
  \begin{pmatrix}1&h\\0&1\end{pmatrix}$.
  Keep $t_x,t_y\sim U[-2,2]$ pixels,
  $a_x,a_y\sim U[-\log1.1,\log1.1]$, and $h\sim U[-.1,.1]$.
- Check the transformed corners of $[-9.5,9.5]^2$ against $[-13,13]^2$,
  including the interpolation footprint. Reject geometrically invalid
  proposals before rendering. Keep the same rule for every role and method.
- Predict $y=(T(z_1)-z_1,T(z_2)-z_2,T(z_3)-z_3)\in\mathbb R^6$ for
  $z_1=(-6,-4),z_2=(6,-4),z_3=(0,8)$, in the existing component order.
  Landmarks are not visible markers. Do not clip predictions before scoring.
- Preserve fixed empirical digit proportions, forward/inverse-map conventions,
  zero padding and pixel units. There is no changing blur or rare-branch draw.

Draw a different angle for each observation from

$$
\alpha_t\sim\operatorname{TN}_{[-50^\circ,50^\circ]}
\bigl(\mu_t,(6^\circ)^2\bigr),
\qquad \mu_t=s\,\mu_{\max}h_t,
$$

where $s\in\{-1,+1\}$ is a preassigned replica-level direction, balanced
within development and confirmation. Initially $\mu_{\max}=25^\circ$.
Use three $h=0,1,0$ holds of 352 incoming batches, separated by 128-batch
half-cosine rise/return ramps with zero endpoint slope. Each batch contains
four new pairs. Pure stationary controls keep $h=0$ for the same full horizon.
Pretraining uses $h=0$ and the same nontrivial regression task and nuisance law.
There is no classification pretraining, head swap, or identity-only warm start.

Continuity refers to the sampling distribution, not to adjacent individual
images. Fixed-support truncated normals and the fixed geometric acceptance
rule define the actual law. Plot proposed and accepted angle densities at
$h=0,.5,1$, acceptance fractions and landmark covariance. Rejection can couple
the nuisance factors to angle; report that coupling rather than calling the
accepted law a product distribution. Never expose $s,h_t,\mu_t$, the true
affine parameters, digit IDs or regime boundaries to the controller. Regression
targets are used only after the corresponding action has been chosen.

Freeze length factors $1$ and $1.25$ before learning. If the timing/settling
pilot demonstrates unresolved carryover through an entire hold, one common
doubling of hold durations is available before confirmation, with a revised
forecast. Preserve ramp durations and reduce cohort size before compromising
hold duration. A nominal five-half-life rule for fixed $.025$ is not a proof
that every adaptive filter or learner has settled; show the actual diagnostics.

Use separate source roles for pretraining, initialization validation, initial
Fisher, offline capability diagnostics, online development, online confirmation
and official-test evaluation. A feasible allocation for the six training roles is
24,000/1,000/3,000/2,000/15,000/15,000 sources respectively. Pretraining is a
stand-in for an available pretrained model; its labels and compute must remain
visible, separately from online efficiency. Use 1,024 initial Fisher inputs,
with an optional nested 2,048-input adequacy check within the reference role.
Noise calibration may use 2,048 already-seen pretraining examples, not new labels.

Partition official test sources into development/confirmation pools before
selection. Within each replica, no online source repeats or crosses a training
role. Replicas may reuse benchmark sources with independently named draws;
they are repeated benchmark experiments, not independent new datasets. Save
source IDs, transforms, accepted/rejected proposal counts, seeds and stream
hashes. All policies in a replica share the exact initialization, initial
Fisher, four-pair batches, schedule, transform draws and evaluation panels.

### Learner, Fisher And Observer Contract

Use the six-output base CNN from the registration architecture, with no LoRA
factors: two input channels, convolution widths $(4,6)$, SiLU/average pooling
and a six-output linear head. Assert **448 trainable parameters** and a stable
flattening order. All are trainable. Use the same architecture and likelihood
for pretraining and online learning, with no categorical gauge projection.

The fixed working likelihood is

$$
Y\mid x;\theta\sim N(f_\theta(x),4I_6),\qquad
L(x,y;\theta)=\frac{\|f_\theta(x)-y\|^2}{8}+3\log(8\pi).
$$

Sum coordinate NLLs and average across observations. The Gaussian is a working
likelihood, not a claim that the synthetic label residuals follow this model.
With $J=\partial f_\theta(x)/\partial\theta$,

$$
s(x,y;\theta)=\frac{J^T(y-f_\theta(x))}{4},\qquad
\operatorname{diag}(F_x)_i=\frac14\sum_{k=1}^6 J_{ki}^2.
$$

Compute per-output Jacobian rows in bounded chunks. Model-target integration
defines the single likelihood Fisher; residual-weighted observed-gradient
squares are only observer measurements, not substitute Fisher observations.
Keep the representation strictly diagonal. No full eigendecomposition, dense
per-sample Hessian, or second Fisher estimand is introduced.

Choose $\pi_t$ before reading the incoming batch. After arrival, evaluate its
direct Fisher diagonal $z_t$ at the pre-solve parameter $\theta_t$, form

$$
d_t^+=(1-\pi_t)d_t+\pi_t z_t,\qquad
\kappa_t=r\,\operatorname{mean}(d_t^+),
$$

and minimize the batch-mean NLL plus

$$
\frac{1-\pi_t}{2\pi_t}(\theta-\theta_t)^T
\left[\operatorname{diag}(d_t^+)+\kappa_t I\right](\theta-\theta_t).
$$

The first batch receives this same direct blend, explicitly versioning the
historical first-batch exception. Update the exact nominal weight concentration
$q^+=(1-\pi_t)^2q+\pi_t^2/4$ from $q_0=1/n_{\rm init}$, and retain $d_t^+$
for the next arrival. Repeated optimizer passes do not add nominal samples.

Use the inherited diagonally scaled physical-objective L-BFGS scheme, initially
50 iterations/75 closures, with its evaluated-best-iterate safeguard. Audit a
small paired set of frozen solves at doubled budget before interpreting policy
differences; apply any necessary solver repair to all arms. Record physical
gradient norms, objective components and stopping reasons. Numerical scaling
floors are distinct from statistical ridge; $r=0$ leaves diagonal EWC active.

Reuse Plan 17's general-purpose score-anchor/Holt/Kalman construction, replacing
only the categorical score/Fisher computations with the formulas above. Start
with a fixed initialization anchor, as in the repaired digit-9 implementation;
do not restart the known problematic periodic reanchoring policy. Keep float64
score moments/filter calculations, noise half-life 32 batches, cold start 16,
and ordinary action limits $[10^{-4},.2]$. Full-initialization noise calibration
is not needed if a representative 2,048-example calibration is adequate; audit
its stability on a nested already-seen subset.

The drift estimate remains latent parameter movement from common-anchor scores.
Do not substitute known angles, angle differences, output-label means or
hand-picked transformation coordinates for it. No schedule-informed policy is
part of the primary experiment. The mean-motion and uncertainty terms in
$S_t$ remain separately recorded. A calibrated innovation score does not prove
accurate drift estimates or optimal learning actions.

### Conditions And Retention Comparability

The principal comparison retains fixed $\pi=.025$. Do not replace it with a
smaller fixed action selected after seeing adaptive results.

| Condition | Adaptation weight | Ridge | Purpose |
| --- | --- | --- | --- |
| Fixed reference | $.025$ throughout | Common $r_0$ | Main practical benchmark |
| Covariance only | Tracked-$q$ formula | $r_0$ | Value of movement estimation |
| Anchor EMA | Selected Holt observer | $r_0$ | Simpler generic observer |
| Anchor Kalman | Selected Kalman observer | $r_0$ | Primary adaptive policy |
| Frozen initialization | No learning | None | Available learning opportunity |

Common-ridge contrasts isolate the policy package. Old/new trade-offs remain
visible; a new-domain gain purchased by materially greater forgetting is not
an acquisition gain at comparable retention.

To avoid repeating Phase 5's ambiguity, allow a **bounded secondary retention
calibration of Kalman** after its observer setting and $r_0$ are frozen. Use a
development band of $\pm0.10$ pixels in the old-domain budget RMS difference
from fixed $.025$; this is a calibration tolerance, not a success threshold or
equivalence conclusion. If common-ridge Kalman already lies in the band, or
has lower mean acquisition error and lower mean old error, retain it without
forcing it to forget more.

Otherwise test at most two additional ridge settings on the same development
replicas: the nearest lower and upper neighbors of $r_0$ in
$\{0,.01,.03,.1,.3,1\}$; at zero use $.01,.1$. Among measured candidates in
the band, select the smallest mean acquisition risk, then smaller absolute old
gap, then lower measured learning cost. If none matches, retain the closest
old gap as explicitly unmatched. There is no interpolation or third search.

If the selected $r_K$ differs from $r_0$, fresh replication adds both
**Kalman at $r_K$ and fixed $\pi=.025$ at $r_K$**, retaining the four common-ridge arms.
This companion control distinguishes a ridge effect from an adaptive-policy
effect at the same ridge. The cross-ridge calibrated comparison is a package
comparison. Use one selected map for changing and stationary schedules, without
retuning per regime. Never filter fresh replicas by realized retention, force
matching after confirmation, or equate a nonsignificant gap with equivalence.

### Phase 6A: Implementation And Recovery Acceptance

Implement in a separately versioned package, proposed as
`mnist_experiment/kalman_pi/registration/`, and new roots under
`cache/mnist_experiment/plan17/registration_<revision>/`. Historical source
manifests and completed units must remain valid. Reuse compatible numerical,
rendering and lifecycle helpers with explicit dependency hashes; do not adapt
the sealed categorical runner in place. Publish the pending notebook section
and live progress as soon as acceptance begins.

Required focused tests cover rendering/angle conventions and continuity,
source/stream pairing, regression pretraining without a head swap, exact
448-parameter ordering, NLL/score signs, analytic model-label Fisher diagonals,
scalar risk/Kalman parity, action-before-batch timing, q bookkeeping and no
schedule leakage. Test the common-ridge and supplemental companion contrasts.
Use float64 finite-difference score checks on tiny states, not large Hessians.

Run tiny CPU and CUDA end-to-end acceptance, including a real process stop
after an accepted batch, exact continuation of learning/filter/RNG state,
immutable-unit collision checks, preserved historical notebook output, resource
caps and a completed-run resume that does no training. Computational tests
count against the phase allowance. These checks validate the new pipeline;
there is no need to rerun every historical integration.

### Phase 6B: Task Readiness And Measured Resources

Use two development replicas with opposite rotation signs. Pretrain on nominal
registration for 15 epochs, with one common extension to 30 if initialization
has not learned the task. Start Adam at $.001$, batch size 256, and show
validation curves plus zero-displacement and constant-mean prediction baselines.
Do not widen the network, change labels, or lower Gaussian variance to make
the observer signal appear stronger.

Evaluate frozen nominal and shifted errors at $\mu_{\max}=25^\circ$. Run short,
separately labeled capability fits with 256 shifted training pairs per replica,
at most 20 epochs, and the same architecture. These diagnostic models neither
initialize production nor contribute fresh evidence. Show whether the shifted
deficit exists and whether additional data can reduce it.

If the deficit is negligible, inspect $35^\circ$; if it exists but the capability
fit does not recover it or geometric rejection is excessive, inspect
$15^\circ$. At most one alternative is evaluated. Select a usable, recoverable
shift on development evidence before examining adaptive-versus-fixed efficacy.
Record the numerical comparisons and selection reason. An unattended default
keeps $25^\circ$ when its shifted frozen MSE exceeds nominal by at least 5% and
the capability fit reduces shifted MSE by at least 5%; these are task-readiness
flags only. If neither tested angle meets both flags, choose the finite
candidate with the largest recovered fraction of its positive deficit, or the
default if neither has a positive deficit. Continue a bounded learning study
with an explicit weak-opportunity qualification rather than abandoning the idea
because a heuristic readiness flag failed.

Run full fixed-$\pi$ development paths at common-ridge candidates
$r\in\{0,.01,.1\}$. Select $r_0$ using the smallest across-replica mean of
$\tfrac12(A^{\rm old}/R^{\rm old}(0)+A^{\rm shifted}/R^{\rm shifted}(0))$;
exact ties prefer lower learning cost, then smaller ridge. A zero frozen
denominator requires a recorded absolute-risk ranking instead. This heuristic
selects a reasonable operating point for the strong fixed reference; it is
not the headline scientific metric. Keep the entire development trade-off plot.

Time complete trajectories, initialization, per-output score/Fisher work,
filtering, solver closures, holdout inference, checkpoint I/O and reporting.
Measure peak VRAM/RAM and bytes per path. Update the forecast before observer
expansion. Do not assume the categorical runtime or Plan 15's FD compute ratio
transfers to this learner.

### Phase 6C: Observer Development And Frozen Selection

Process the same development score sequences in shadow using
$H\in\{8,16,32\}$ and $\eta\in\{10^{-5},10^{-4},10^{-3}\}$.
Use the original regime-balanced predictive innovation log-score ranking,
displaying calibration and failures, to retain at most two candidates per
observer family. Run those candidates on-policy on both development replicas
at $r_0$, with the covariance-only and fixed controls. Choose one EMA and one
Kalman setting using the same transparent normalized old/shifted-risk ranking
as 6B; retain both outcome coordinates and all unfavorable candidates.

Only one adjacent half/double or decade expansion per family is allowed when
a concrete lag, saturation or numerical diagnosis warrants it. Do not tune
Kalman gain directly, remove uncertainty silently, add an IMM factorial, or
use holdout errors as an online feedback controller. A necessary mathematical
repair receives a new protocol and explicit notebook explanation; benign
continuation does not require the user to switch models.

Before freezing selection, make a small independent audit of fixed-anchor
score means at $h=0,.5,1$ using at most 256 additional diagnostic-role pairs
per state and replica. These are charged offline diagnostics, never inputs to
live policies or Fisher initialization. Compare measured mean-score changes
and their uncertainty with ordinary batch noise; show filter innovation,
lag, mean-motion and uncertainty on the common paths. These observations
measure score response, not the true parameter path or a drift oracle.

Run the bounded retention calibration above if needed. Save an immutable
decision containing task/angle law, horizons, initialization, solver settings,
common ridge, observer parameters, optional calibrated ridge, every candidate,
selection reasons, development IDs and forecast. Freeze confirmation seeds,
arm list and block counts before any confirmation outcome is exposed. No
numerical benefit or significance threshold is required to proceed.

### Phase 6D: Fresh Paired Exploration

Target **eight fresh changing replicas and four fresh stationary replicas**,
balanced across signs and length factors, with the four common-ridge policies:
**48 full trajectories**. If the secondary ridge differs, its two companion
arms add 24 trajectories, giving **72 at most**. The stationary arm retains
its assigned sign for evaluating the same shifted holdout, although its training
distribution never leaves nominal. Initializations are fresh per replica and
shared exactly across its policies.

Use measured costs plus 25% remaining-run contingency and reserve at least
one active hour for analysis/closeout. Before confirmation starts, omit
unstarted optional observer expansions first. If needed, reduce to four changing
and four stationary blocks balanced across all four sign/length combinations.
Do not drop a ridge companion while keeping its Kalman arm, shorten stable
periods to manufacture replication, or select the allocation by p-values.
If even that cohort cannot fit, retain the evidence and report the resource
limit; do not silently increase the allocation.

Interleave complete paired blocks and both schedules. Partial blocks are
visible progress but excluded from paired contrasts until complete. A node
interruption may leave them pending; `--resume` finishes the identical design.
Fresh outcomes never select new settings or additional replicas. Weak or
contrary findings are reported and the frozen cohort is completed within budget.

### Metrics And Human-Friendly Phase 6E Report

At initialization, every 16 accepted batches and every regime boundary,
evaluate independent, fixed **nominal** and **fully shifted** holdout panels,
100 sources per digit, with fixed empirical digit weights. Use paired sources
across panels with separately recorded transforms. The shifted panel uses the
selected replica sign and $h=1$ at every checkpoint, even during the return;
it does not become easier as the training distribution returns to nominal.
Optionally show a smaller current-law panel, clearly separate from these two
fixed targets. Only development-panel summaries select settings. Confirmation
panels never train, supply scores/Fishers, choose actions or retune a recipe.

For policy $a$, replica $r$, and processed pair count $n$, define

$$
R^D_{a,r}(n)=\mathbb E_{D,\rm holdout}
\left[\frac13\sum_{k=1}^3
\|\widehat T_{a,r,n}(z_k)-T(z_k)\|^2\right],
\qquad D\in\{\mathrm{old},\mathrm{shifted}\}.
$$

The square root is RMS landmark error in **pixels**; regularizers never enter
this risk. Also show the percentage of landmarks within one pixel, fixed
before development, with per-digit/coordinate results in linked diagnostics.
Count one reference/query pair as one example and two image views. Six outputs
are not six independent samples. Include pretraining and diagnostic labels
separately from ordinary online exposure.

Let $A^{\rm old}$ be the observation-budget average of $R^{\rm old}$ over the
whole path and $A^{\rm shifted}$ the average of $R^{\rm shifted}$ over the
rise plus high hold, when acquisition is relevant. Calculate normalized
trapezoidal areas on the saved actual-arrival coordinates, including exact
segment boundaries. For stationary controls, use the same preassigned time
window without implying an actual rise. Also report whole-path shifted risk
and every regime's risks so window choice cannot conceal deterioration.

Report paired gains $A^D_{\rm fixed}-A^D_a$ in square pixels, positive meaning
better, alongside budget RMS gaps
$\sqrt{A^D_a}-\sqrt{A^D_{\rm fixed}}$ in pixels, negative meaning better.
The latter old-domain gap defines the development matching band. Show
forgetting from initialization, endpoint recovery and post-return shifted-task
retention separately. Take roots within replicas before averaging gaps.

Use 2,000 named, persisted whole-replica paired bootstrap draws, shared across
endpoints and contrasts, stratified by sign/length block, with changing and
stationary cohorts separate. If a reduced cohort has only one replica per
stratum, use the ordinary paired-replica bootstrap and explicitly state that
between-stratum variation enters uncertainty. Report all adaptive-versus-fixed
common-ridge contrasts; covariance and EMA comparisons are secondary. If
present, report both matched-ridge contrasts and the fixed-policy ridge contrast.
No checkpoint-level pseudoreplication, discarded unfavorable replicas, or
equivalence claim from an interval crossing zero.

The result concerns predictive quality per equal incoming-data budget and its
compute cost. Do not claim actual samples saved or manufacture an $S$ statistic
by inverting nonstationary curves: earlier/later checkpoints also encounter
different distributions. A separate sample-equivalence design would be needed.

Append **Phase 6: Adaptation Weights For Single-Digit Registration** after the
Phase 5E conclusion in `plan17_results.ipynb`. Begin with a short version of
the motivation and hypothesis above. Keep the main addition to four multi-panel
figures and two small tables:

1. Representative reference/query images with landmark overlays, angle-law
   densities and the hold/rise/hold/return/hold schedule. Show a frozen-model
   readiness summary, including any unsuccessful context probe.
2. Development old-versus-shifted error frontier, selected common/optional
   calibrated ridges, and a small observer-health panel.
3. Fresh old/shifted RMS error and within-one-pixel accuracy, with frozen
   references, complete-pair counts, regime bands and matching axes.
4. Aligned adaptation weights and mean-motion/uncertainty, plus paired applied
   effects with nominal 95% intervals. Link fuller regime diagnostics.

Use one settings table and one conclusion table with both errors, intervals,
retention-matching status and learning/online-wall cost ratios. Explain whether
the evidence shows useful adaptation, a trade-off, mostly conservative weighting,
or uncertainty. Generic sensitivity to a changed image distribution is not
proof of accurate $\theta^\star$ tracking. No oracle-timing or shuffled-action
experiment means any benefit of varying weights is a policy-package result,
not definitive identification of the value of its temporal alignment.

Update near-top progress with phase/revision, completed/planned paired blocks,
active checkpoint/heartbeat, compute used/allocation, forecast remaining and
storage. Preserve historical sections/outputs and sealed report generations;
do not regenerate them on every arrival. Refresh atomically at stage boundaries,
graceful stops and at most every 15 active minutes. The notebook loads artifacts
only and clearly distinguishes pending, partial, incompatible and completed work.

### Phase 6 Resources, Autonomy And Recovery

Execution authorization on 2026-11-04 grants a **new, separately authorized
maximum of 20 active wall-clock hours for Phase 6**. The previous 14.318 hours remain recorded;
its roughly 5.68-hour unused allowance is not silently reset or transferred.
Under this proposal, the total spent through Phase 6 would be at most about
34.32 hours. Preserve separate phase and cumulative
family totals so restarts cannot renew either allocation.

| Sub-phase | Provisional active compute | Deliverable |
| --- | ---: | --- |
| 6A: implementation acceptance | 0.5--1 h | Tested regression, pairing, recovery, live section |
| 6B: readiness and timing | 1.5--3 h | Learnability/context checks, fixed-ridge screen, measured forecast |
| 6C: observer and retention development | 2--4 h | Diagnostics and frozen settings/cohort |
| 6D: fresh paired exploration | 4--8 h | 48 paths, or at most 72 with ridge companions |
| 6E: analysis and closeout | 0.5--1 h | Concise applied conclusions and full artifact audit |
| **Planning range** | **8.5--17 h** | Approximately **10--18 h**, subject to measured timing |
| **Authorized hard allowance** | **20 h** | Includes tests, diagnostics, retries, reports and I/O |

Initial programming is estimated at **4--8 engineering hours** separately;
computational tests are included above. These are forecasts from related small
models, not a measured registration-controller benchmark. The 25% contingency
must fit inside the cap, so all upper ranges cannot automatically be spent.
After authorization, Sol may make documented correctness repairs and the bounded
development decisions above without waiting for Astra or the user. Preserve
invalidated evidence with its explanation and use new identities for material
changes. New tasks, expanded factorials, more confirmation for significance,
or an increased resource cap require a new agreement.

Retain the shared **500 GB peak-artifact ceiling**, at least **20 GB actual
free disk**, and a provisional **under-10-GB new-data target**. Use source IDs
and immutable transform records with deterministic rendering; cache only what
the measured runtime requires. Save compact scalar metrics and bounded rotating
recovery checkpoints, not every full model after every arrival. Completed
trajectories and final scientific decisions remain immutable and checksummed.

Require `--resume`, bounded per-invocation time/unit limits, status/stop commands,
signal and stop-file handling, and one writer per study. Checkpoint model,
Fisher, q, filter/noise/anchor state, pending data/renderer state, RNGs, accepted
metric offsets and charged time atomically at each accepted arrival; bound
pretraining/diagnostic loss to one recorded chunk. Test interruption both during
an arrival and after its commit. Resume must validate identities, preserve all
completed bytes and continue without duplicate assimilation or labels. Check
stop requests inside solves, reference chunks, validation and reporting; target
a graceful stop within one minute and document any unavoidable longer kernel.

The scheduler must progress unattended through authorized sub-phases and sealed
decisions, including after credit exhaustion or client disconnection. A node
restart requires the documented resume command, not a new design decision.
Record verified commands in `PLAN17_HANDOFF.md` after implementation. Local
execution needs no tmux; use it if execution is later requested remotely.
Finish with an updated notebook, resource ledger, bounded full audit and clear
handoff. Do not launch any subsequent study automatically.

### Phase 6A Implementation And CPU Acceptance Record

Implemented on 2026-11-04 under `kalman_pi/registration/`. The combined CPU
acceptance suite passes **101 tests**; final learner/runtime/pretraining/readiness
checks pass **13 tests** after the last small additions. Coverage includes
Gaussian scores/Fisher diagonals, deterministic role/stream pairing, causal
actions and the first-batch blend, independent resource accounting, per-batch
pretraining recovery, actual process death with exact continuation, immutable
selection and completed units, bootstrap blocks, and historical notebook safety.

The final source revision's CPU CLI smoke root is
`cache/mnist_experiment/plan17/registration_acceptance_cpu_v3`: an intentional
six-unit stop and resume completed **41 sealed units / 18 tiny trajectories**,
all fully checksum-validated. These are implementation tests, not development
or confirmation evidence. Earlier CPU v1/v2 roots are preserved as prototype
acceptance under their prior source identities and must not be resumed or pooled.
The canonical notebook now appends a pending Phase 6 section and progress panel;
all 39 historical cells and inherited metadata are unchanged. Older publishers
cannot remove the new section. Completed digit-9 scientific sources remain intact.

Phase 6 has charged approximately **0.11 active hours** so far, with exact totals
in its acceptance ledger and smoke-root runtimes. Historical spending remains
**14.318 hours**. GPU acceptance is still required: no NVIDIA driver module was
loaded at kickoff, despite an installed matching module. The user will update
the driver, reboot and confirm `nvidia-smi`. Per that explicit instruction,
**no production readiness fits, calibration or fresh experiments have started**.
The production root contains only report/status artifacts, without a scientific
run identity or learning units. Verified CPU and deferred CUDA/production commands
are recorded in `PLAN17_HANDOFF.md` for the next session.

### Phase 6A Post-Reboot Acceptance

After the forced restart on 2026-11-04, a read-only audit verified all **41**
completed CPU units, loaded **62** checkpoint files, and matched all **31**
scientific source hashes. The canonical notebook matched its sealed generation;
all **39** historical cells and inherited metadata remained unchanged. No
production learning had started. The CPU suite passed again: **101 tests**.

With the repaired driver, PyTorch **2.10.0+cu128** can use the RTX 4070.
The separate CUDA smoke study, `registration_acceptance_cuda_v1`, stopped after
six sealed units, resumed to **41 units / 18 tiny paths**, and passed full
checksum/source/report validation. All six earlier completion seals remained
identical. These tiny paths are acceptance evidence only, never applied results.
The integrity record is `phase6A_post_reboot_integrity.json` in the Plan 17
cache. Acceptance remains charged to the Phase 6 ledger, without changing the
historical 14.318-hour total. The user's conditional authorization to continue
is satisfied; production resumes in `registration_v1`.

### Phase 6 Reporting Recovery Correction

During Phase 6C/6D transition, a parallel read-only review found a cross-layer
reporting defect not covered by the original acceptance tests. A live metrics
list can include a row whose checkpoint transaction was interrupted; scheduler
startup also temporarily omits the active partial trajectory. The report's
strict prefix rule could then reject an otherwise valid scientific resume.

Production was gracefully checkpointed before the first fresh path completed.
The reporting-only correction labels incomplete previews provisional and permits
their disappearance or metric rollback/replay. Their identities remain checked;
completed trajectories, settings, source identities, history and compute totals
retain strict validation. No learner, observer, selection or scientific source
changed. A real scheduler fault-injection test interrupts checkpoint publication,
publishes the provisional row, and resumes successfully from the earlier durable
state while preserving all earlier seals. Separate regression tests cover the
publisher guards. Group frozen-reference plot lines now use the group initial
mean, and the retention-band text is derived from the frozen development record.
Earlier report generations are retained; incomplete paths never enter inference.
The final combined acceptance suite passes **154 tests**, including the new
full-scheduler recovery test. All **31** pre-stop completion seals and all
**31** scientific source hashes are unchanged; production resumes at its durable
accepted step 98. A conservative 240-second test/report charge is retained in
the Phase 6 ledger.

### Phase 6B--6E Execution And Conclusion

Both development readiness probes passed at the original 25-degree context.
Frozen nominal/shifted/capability-fitted shifted RMS was .741/1.989/1.183 pixels
for replica 0 and .761/2.075/1.301 for replica 1. No alternative context was
needed. Four doubled-budget frozen solver checks found zero objective gap.

The immutable selection froze before fresh outcomes: common ridge **.01**,
fixed adaptation weight **.025**, **EMA H=8**, **Kalman eta=.0001**, eight changing
and four stationary replicas, four common-ridge policies. Kalman's development
old RMS gap was -.00432 pixels, within the +/- .10 calibration band; no secondary
ridge companions were required. All 16 development and 48 fresh paths finished.

Changing-schedule paired RMS gaps below are policy minus fixed, in pixels;
negative favors the adaptive policy. These are nominal whole-replica 95%
bootstrap intervals, not multiplicity-adjusted claims.

| Policy | Old RMS gap [95% CI] | Shifted RMS gap [95% CI] |
| --- | ---: | ---: |
| Covariance only | -.0132 [-.0167, -.0097] | +.0563 [.0484, .0642] |
| Anchor EMA | -.0088 [-.0123, -.0052] | +.0377 [.0298, .0455] |
| Anchor Kalman | -.0013 [-.0064, .0039] | +.0088 [.0033, .0143] |

**No joint applied improvement is established.** Fixed .025 has a small but
nominally supported shifted-error advantage over Kalman; a Kalman old-domain
benefit is not established. Covariance and EMA buy retention with larger
shifted-error costs. On stationary controls, Kalman's old gap was -.0028
[-.0058, .0002] and shifted gap +.0598 [.0395, .0802]. The latter concerns
untrained-domain generalization, not acquisition without shifted arrivals.
A zero-spanning interval does not establish equivalence. These findings concern
this Gaussian registration learner, not a universal verdict on adaptive weights.

Kalman used **.865x [.840, .891]** learning time and **.970x [.961, .979]**
online wall time on changing schedules. It averaged 13.50 solver closures per
arrival versus 18.56 for fixed. Same solver caps were used; its stronger typical
constraint reduced solve work enough to offset observer overhead. This was an
equal-arrival-budget comparison, not a compute-matched superiority test.
No actual-samples-saved or S statistic is claimed for these nonstationary curves.

The planned nested noise audit reused 1,024 already-seen examples against the
sealed 2,048-example calibration: raw variance-trace ratios .9946/1.0044 and
Fisher-weighted noise ratios .9966/1.0182. The nested reference is correlated,
not truth; these are consistency diagnostics, not proof of observer calibration.
Saved solver/score audits were restored from their immutable units after a
resume omitted them from the live summary. This reporting repair did not change
selection or learning. Applied-weight plots now omit only the unavailable
initial action, and pool directions within the same length/exact grid for
readability, without interpolation. Small-effect intervals retain four decimals.

Final checks passed **158 unit tests and six integration tests**, including
real process death and checkpoint-abort/publication/resume. Full checksum/source
validation passed **113 production units**, 41 current CPU acceptance units,
41 CUDA acceptance units and all 356 historical production units: **551 total**.
All 31 pre-repair seals, all 39 inherited notebook cells/metadata and all
scientific source hashes remain intact. The notebook contains 15 nonblank PNGs.
Completed production resume returned `already_completed` without training.

Spending is approximately **4.30 / 20 Phase 6 hours**, independently of the
preserved **14.318 historical hours** (about **18.62 total**). Production artifacts
occupy about .67 GB, far below the shared 500 GB ceiling. No worker remains.
The final report is `registration_v1/report/generation-71e194b6c26aaaa7be639182`;
final audit/noise artifacts and ledgers are retained under the same study/family
roots. `kalman_pi.registration.acceptance --closeout --root <root>` performs the
bounded completed-artifact audit and publication; its small already-seen-data
noise check is cached. No subsequent experiment is started automatically.

## Phase 7: Discrepancy-Informed Adaptation During Sustained Drift

### Motivation And Research Question

The long-term objective remains statistically and computationally efficient
continual learning with limited observations and compute. This phase focuses
on the mechanism: understand when an adaptation weight informed by signal and
noise actually helps a learner follow a changing population. Demonstrating a
deployable resource-allocation policy or simultaneous improvement in every
retention outcome is not required at this stage.

Phase 6 already used continuous rotation-distribution changes, not binary task
switches. Its half-cosine ramps occupied only about 20% of incoming observations;
the remaining time consisted of three stationary holds. Observer actions reacted
to the ramps, but the changing-schedule Kalman condition had shifted RMS error
0.0088 pixels higher than fixed .025, with nominal 95% interval [0.0033, 0.0143].
Its nominal-domain retention benefit was unresolved. Sensitivity to change was
therefore visible without an established joint predictive improvement.

The discussion identified three reasons for a more informative follow-up:

1. Longer uninterrupted drift gives a causal observer more time to detect and
   act on movement. Remove the middle hold and allocate its observations to
   the outward and return ramps, retaining quiet periods before and after.
2. The current velocity-based rule estimates population movement, whereas the
   actual learner may also have accumulated tracking error. The conditional
   target in [the mathematical overview](../mathematical_overview.ipynb) uses
   the discrepancy between the next population solution and the current
   learner. The existing level/velocity filter offers a causal way to estimate
   that target, with explicitly modeled uncertainty.
3. Fixed endpoint holdouts describe acquisition and retention but do not measure
   performance on intermediate distributions. Add independent current-law
   holdouts as the primary tracking outcome and keep endpoint outcomes visible.

Compare 25-degree and 45-degree excursions. Increasing amplitude may improve
detectability, but may also challenge the fixed-anchor approximation and model
capacity. Padding is a bounded engineering task, not a reason to discard the
larger excursion. Fix the rendering and sampling support before calibration.

Smooth environmental schedules do not certify LAN, accurate score-to-parameter
mapping, or the matched-quadratic EWC approximation. The overview's local
scaling concerns statistical parameter distance, not a universal degrees-per-
batch limit. Slower drift can improve locality while reducing per-batch signal.
Treat this balance as the hypothesis, not as an explanation already established
by Phase 6. Historical [Plan 7](plan7.md) already examined conditional anchor
error, and [Plan 10](plan10.md) found reliable finite-EWC solves without the
predicted pace/weight relationship. No new oracle-existence or candidate-pi
response-surface study is included here.

> **Question:** during sustained, gradual registration drift, does a causal
> estimate of remaining target-to-learner discrepancy improve current-law
> prediction relative to fixed .025, covariance-only adaptation, and the
> existing velocity-based Kalman recommendation?

**Status:** Phases 7A--7E are complete in `discrepancy_v1`: all 40 development
and 80 fresh trajectories are sealed. Final acceptance passed 81 tests; full
artifact validation and completed-resume checks passed. Charged Phase 7 compute
is **9.34 / 20 active hours**, including closeout. No worker or follow-on study
is running.
Astra supplied the design. Sol implements, calibrates, executes
and closes the phase without requiring
intermediate model switches or user reviews. Preserve all earlier evidence.

### Task And Padded Rendering

Keep one reference/query pair per observation, three virtual landmarks, six
pixel-displacement outputs, all ten digit classes at fixed empirical proportions,
and the same affine conventions and nuisance ranges as Phase 6. Keep the
antialiased digit itself at 18 by 18 pixels. Enlarge both canvases to **40 by
40 pixels before warping**, with black padding; padding a cropped query after
warping cannot recover lost content. Keep landmark coordinates relative to the
canvas center at $(-6,-4),(6,-4),(0,8)$ and retain physical pixel units.

For the existing support square, scale, shear and translation bounds, a
conservative bound on either transformed coordinate, valid at every rotation,
is

$$
b_{\rm support}
\le 2+9.5(1.1)\sqrt{(1+.1)^2+1}
\approx17.535\ \text{pixels}.
$$

Use a verified support allowance of 19 pixels on the 40-pixel canvas. Validate
the interpolation footprint, center convention and inverse sampling grid in
7A. This envelope should allow every configured affine proposal. Replace
angle-dependent geometric resampling with an assertion of valid support; an
unexpected failure is a renderer defect to resolve, not permission to alter the
law silently. Retain proposal/acceptance diagnostics so this claim is testable.

For every data role and condition use

$$
\alpha_j\sim\operatorname{TN}_{[-70^\circ,70^\circ]}
\bigl(\mu_j,(6^\circ)^2\bigr),\qquad
\mu_j=s\,a\,h_j,\quad a\in\{25^\circ,45^\circ\},
$$

with independent uniform proposals for the unchanged translation, log-scale
and shear factors. At the largest location, the untruncated angle mass outside
the support is approximately $1.55\times10^{-5}$. The 45-degree designation
is the maximum magnitude of the distribution's location, not a hard bound on
individual sampled angles. Plot the actual truncated density, sampled angle
histograms and representative images for both excursions, including endpoints.

Use the same two-convolution widths $(4,6)$, SiLU, initial average pooling and
six-output head as Phase 6, with a final adaptive average pool to $2\times2$.
On the fixed $20\times20$ intermediate map, implement this operator as
`AvgPool2d(10)` to retain deterministic CUDA backward; its values and gradients
agree with the adaptive operator within floating-point rounding.
Assert **448 trainable parameters** and an unchanged output coordinate order.
Train initialization anew on the padded nominal regression task. No old
checkpoint, classification warm start, head swap, extra STM, replay, FD KL,
LFU or dense Fisher condition is introduced. Padding/pooling and removal of
geometric rejection change the context; Phase 6 is historical evidence, not a
paired control isolating only the new schedule or discrepancy formula.

### Schedule, Pairing And Data Roles

At base length, preserve **1,312 incoming batches, four pairs per batch, and
5,248 incoming pairs**. Use the following checkpoints in accepted-batch units:

| Segment | Batch boundaries | Pair boundaries | Rotation location |
| --- | ---: | ---: | --- |
| Initial hold | 0--352 | 0--1,408 | 0 |
| Outward ramp | 352--656 | 1,408--2,624 | $0\to s a$ |
| Return ramp | 656--960 | 2,624--3,840 | $s a\to0$ |
| Final hold | 960--1,312 | 3,840--5,248 | 0 |

For each 304-batch ramp, use the existing half-cosine sampling, including both
endpoints, and reverse it for the return. There is no intervening plateau;
the joined path has zero velocity at the turning point. Approximately 46% of
arrivals now lie on ramps. Maximum adjacent location changes are approximately
.130 degrees at amplitude 25 and .233 degrees at amplitude 45, versus .309
degrees in the original base schedule. These are environmental diagnostics,
not asserted Fisher-distance bounds.

Retain preassigned confirmation length factors $L\in\{1,1.25\}$, scaling all
segments and using integer ceiling as in Phase 6. The longer path has holds
of 440 and ramps of 380 batches, totaling 1,640 batches / 6,560 pairs. Balance
rotation signs and length factors before learning. Development uses opposite
signs at base length. Stationary controls have the identical assigned horizons
but $\mu_j=0$ throughout. Do not shorten holds or change ramp lengths after
seeing efficacy, and do not pass the schedule, angle, digit identity or segment
boundaries to an action policy.

Reuse the Phase 6 disjoint source-role partition contract, with new Phase 7
component seeds and immutable assets: initialization, initialization validation,
reference Fisher, offline diagnostics, online development, online confirmation,
and disjoint development/fresh official-test pools. The 24,000/1,000/3,000/2,000/
15,000/15,000 training-role allocation remains feasible. Use 1,024 initial
Fisher inputs and a 2,048-example already-seen initialization noise subset.
Record pretraining labels/epochs separately from new online exposure.

Within each replica, all four policies share initialization, initial Fisher,
source order, transforms, optimizer settings and evaluation panels. Pair both
excursions using the same initialization, sources and underlying transform
uniforms, with the angle location changed as assigned. No online source repeats
within a trajectory. Reuse across paired excursions is intentional pairing,
not additional independent evidence. Distinct replicas are independently seeded
benchmark repeats, not independent new datasets.

### Conditional Discrepancy And Optimal Weight

Use the applied fixed-batch model from the mathematical overview. With current
learner error $e_t=\widehat\theta_t-\theta_t^\star$ and population movement
$d\theta_t=\theta_{t+1}^\star-\theta_t^\star$, define

$$
\delta_t=d\theta_t-e_t=\theta_{t+1}^\star-\widehat\theta_t.
$$

For a hypothetical new-batch estimate with error $\epsilon_{t+1}$, matched
local quadratic curvature gives

$$
e_{t+1}\approx-(1-\pi_t)\delta_t+\pi_t\epsilon_{t+1}.
$$

Freeze the predictable risk metric $G_t=\operatorname{diag}(d_t)$ before the
incoming batch. Under the state-space working model, with fresh estimation
noise centered conditional on the latent target and prior history, set

$$
A_t=\mathbb E[\delta_t^TG_t\delta_t\mid\mathcal F_t],\qquad
B_t=\mathbb E[\epsilon_{t+1}^TG_t\epsilon_{t+1}\mid\mathcal F_t].
$$

Conditional centering removes the cross moment. Therefore

$$
R_t(\pi)\approx(1-\pi)^2 A_t+\pi^2 B_t,\qquad
\frac{dR_t}{d\pi}=-2(1-\pi)A_t+2\pi B_t,
\qquad \pi_t^{\rm discrepancy}=\frac{A_t}{A_t+B_t}.
$$

For a known population target, $A_t=\|\delta_t\|_{G_t}^2$, reproducing the
overview's realized-anchor conditional oracle. Averaging over repeated learners
with a fixed comparison metric and population path recovers

$$
\mathbb E\|\delta_t\|_{G_t}^2
=\|d\theta_t-\mu_t^e\|_{G_t}^2
+\operatorname{tr}(G_t\Sigma_t^e),
\quad \mu_t^e=\mathbb E[e_t],\quad\Sigma_t^e=\operatorname{Cov}(e_t),
$$

and hence the marginal numerator when $\Sigma_t^e=q_tK_t^{\rm old}$. These
are different conditioning schemes for the same local risk, not a new Fisher
estimand. For the deployed conditional estimate, do **not** add
$q_tD_t^{\rm old}$ again. The exact scalar $q_t$ recursion remains in effect
for information-weight bookkeeping and for the covariance-only control.

Under a fixed observer anchor $a_0$, retain the existing per-coordinate
level/velocity model $z_t\approx\theta_t^\star-a_0$, with transition
$F=\begin{pmatrix}1&1\\0&1\end{pmatrix}$. Its filtered state uses only batches
already accepted. Before the next batch, predict

$$
\widehat\delta_t=a_0+\widehat z_t+\widehat v_t-\widehat\theta_t,
\qquad
C_{t+1\mid t,ii}=P_{zz,t,i}+2P_{zv,t,i}+P_{vv,t,i}+Q_{zz,i}.
$$

This is the predicted **level** covariance from $FP_tF^T+Q$; it is not the
velocity/displacement covariance used by the existing policy. The conditional
numerator and action are

$$
\widehat A_t=\sum_i d_{t,i}
\left(\widehat\delta_{t,i}^2+C_{t+1\mid t,ii}\right),\qquad
\widehat\pi_t^{\rm discrepancy}
=\frac{\widehat A_t}{\widehat A_t+\widehat B_t}.
$$

Keep the previously specified observed-score map
$y_t=A_{a_0,\epsilon}^{-1}\overline s_t(a_0)$, with $s=-g$, and within-batch
centered score-variance estimates. With the same numerical mapping floor,

$$
K_{t,i}=\frac{\widehat{\operatorname{Var}}_t(s_i(a_0))}
{(A_{a_0,\epsilon,ii})^2},\qquad
V_t=\sum_i d_{t,i}K_{t,i},\qquad \widehat B_t=V_t/m.
$$

Treat this as a calibrated approximation to new-estimate noise, not an identity
for a four-example neural-network MLE. Model-based target uncertainty is also
not a demonstrated frequentist confidence region. No Gaussian assumption is
needed for the quadratic moment identity itself; the interpretation of filter
means/covariances depends on the observer model. Off-diagonal score structure,
fixed-anchor nonlinearity, bias, and unequal old/new curvature can invalidate
the online approximation. Ridge and finite EWC optimization further limit the
scalar-affine derivation. Record these qualifications; do not promise global
optimality, an unbiased discrepancy estimate, or an exact LAN regime.

### Conditions And Shared Learner

| Condition | Unclipped action | Role |
| --- | --- | --- |
| Fixed .025 | $.025$ | Retained strong reference |
| Covariance only | $mq_t/(1+mq_t)$ | Adaptation without an estimated movement term |
| Velocity Kalman | $(S_t+q_tV_t)/(S_t+q_tV_t+V_t/m)$ | Existing movement-based policy |
| Discrepancy Kalman | $\widehat A_t/(\widehat A_t+V_t/m)$ | Conditional tracking-error policy |

For the velocity policy preserve
$S_t=\sum_i d_{t,i}(\widehat v_{t,i}^2+P_{vv,t,i}+Q_{zz,i})$ exactly. Use
separately named implementations so the old policy cannot silently acquire
the new level terms. Frozen initialization is an evaluation reference, not a
fifth trained condition. Do not add EMA, mean-only, oracle, shuffled-action or
alternative-Fisher arms to the primary allocation.

Keep Phase 6's Gaussian likelihood with covariance $4I_6$, integrated-label
diagonal Fisher $\operatorname{diag}(J^TJ)/4$, direct-EMA update including the
first batch, ridge $\kappa_t=r\operatorname{mean}(d_t^+)$, and evaluated-best
L-BFGS solve. Both policies use the pre-arrival $G_t$ in their recommendations;
the common learner then uses its ordinary batch-updated diagonal in EWC. One
chosen action controls both EWC odds and Fisher blending. Do not post-multiply
the accepted optimizer displacement by pi.

For all adaptive arms retain 16 cold-start batches at .025, action limits
$[10^{-4},.2]$, and a documented covariance-only fallback on nonfinite or
unsupported estimates. If both surrogate coefficients vanish, use that same
fallback and flag it. Record unclipped recommendations, clipping and fallback
frequencies. Use float64 for observer moments, Joseph covariance updates and
the level/velocity cross term; log covariance repairs before interpretation.

Begin with a fixed initialization observer anchor and 32-batch measurement-
noise half-life. Do not automatically reinstate the previously unstable
periodic reanchoring scheme. Monitor its limitations as described in 7C.
True angles, current-law holdout losses, population fits and schedule metadata
are unavailable to online policies. Development evaluation may select a frozen
recipe; it is never online controller feedback.

All primary conditions use one common calibrated ridge and the same solver
settings. This phase prioritizes tracking under equal incoming-data budgets;
it does **not** enforce retention matching or add condition-specific ridge
companions. Old-domain gains/losses must be reported so a tracking improvement
is not described as a retention-matched improvement. Independently selected
Kalman process scales make their fresh comparison a calibrated-policy-package
comparison. Log both action formulas in shadow on identical saved observer
states to show the immediate formula difference without claiming a causal
long-run ablation.

### Independent Evaluation And Headline Metrics

Evaluate at initialization, every 16 accepted batches, the exact turning point
and all segment boundaries. Use saved integer incoming-pair counts as the
common horizontal axis. All holdout losses exclude ridge/EWC terms.

For each replica, construct three evaluation targets from its isolated official-
test pool: fixed nominal ($\mu=0$), fixed endpoint ($\mu=s a$), and current law
($\mu=\mu_j$ of the most recently accepted batch). At initialization, current
law is nominal. Current-law evaluation is **post-batch independent holdout
evaluation**, not training-batch loss or prequential prediction. Its angle law,
nuisance law and digit weights match the corresponding incoming distribution.

Use 100 fixed test sources per digit and empirical training-split digit weights.
Reuse frozen source IDs and underlying transform uniforms across checkpoints
and policies; only the declared current-law location changes. This reduces
Monte Carlo variation but does not make time points independent. Persist all
panel seeds, transforms/hashes and role identities. When current and endpoint
panels coincide exactly, reuse predictions rather than rerun inference.
Stationary paths have current law equal to nominal; their 25/45 endpoint
panels measure untrained-domain generalization, not acquisition.
Evaluate the frozen initial model on each current-law panel as well: its
reference curve generally changes with the evaluation distribution and must
not be drawn as a horizontal initialization line. Fixed-panel references can
remain horizontal. Cache these shared frozen predictions once per replica.

Let $R^{D}_{a,r}(n)$ be mean squared landmark distance for policy $a$, replica
$r$, and evaluation target $D$, averaging the three squared Euclidean landmark
errors and then the weighted holdout. Plot $\sqrt{R^D(n)}$ in pixels and the
percentage of landmarks within one pixel. Preserve the Gaussian NLL in linked
tables; it is affine in this squared error and not independent corroboration.

Define the observation-budget average and its RMS summary by

$$
A^D_{a,r}=\frac1{N_r}\operatorname{Trapz}
\left(R^D_{a,r}(n),n\right),\qquad E^D_{a,r}=\sqrt{A^D_{a,r}}.
$$

The primary scalar is **current-law budget RMS** $E^{\rm current}$: the square
root of average squared tracking error over the incoming-pair budget. It is
not the average of checkpoint RMS values. Include exact endpoint checkpoints
in the numerical integration. Report policy-minus-fixed paired gaps in pixels
(negative is better), and velocity-minus-discrepancy gaps with their direction
explicitly labeled. Calculate roots within replicas before averaging effects.

Show whole-path current-law error first, then outward/return ramp and quiet-hold
summaries separately. Report nominal and endpoint budget RMS, initial-to-final
forgetting/recovery, within-one-pixel accuracy and frozen-model references.
No outcome or regime is removed for being unfavorable. Do not require a
significant joint retention/acquisition win. Publish learning-time and online-
wall ratios, observer/evaluation/I/O breakdowns, VRAM/RAM peaks and artifact
bytes. These are equal-data comparisons, not compute-matched claims. Do not
invert these nonstationary curves into samples saved or an S statistic.

### Phase 7A: Implementation, Rendering And Recovery Acceptance

Implement in a new versioned package, proposed as
`mnist_experiment/kalman_pi/discrepancy/`, with new run roots under
`cache/mnist_experiment/plan17/discrepancy_<revision>/`. Reuse frozen compatible
helpers by recorded dependency hash; leave Phase 6 scientific sources and
completed units intact. Initial programming and later execution belong to Sol
after authorization. Publish a pending chronological notebook section and
near-top progress immediately when implementation starts.

Acceptance covers padded support at extreme allowed transforms, identity and
rotation conventions, nonblank rendering without clipping, density/truncation
checks, fixed parameter count, smooth schedule endpoints, data-role pairing,
score signs and likelihood Fisher, discrepancy/velocity action distinction,
the full $P_{zz}+2P_{zv}+P_{vv}+Q_{zz}$ prediction, absence of an extra q term,
and the marginal/conditional quadratic identities. Include stationary targets
with nonzero learner error and reversal cases so velocity-only substitution
cannot pass. Tiny algebraic moment tests are not a revived synthetic LFU study.

Test action-before-batch timing, actual-learner input to discrepancy, q/EMA
bookkeeping, independent current-law panels, paired analysis and current/endpoint
panel deduplication. Run tiny CPU and CUDA end-to-end tests, including a real
process interruption, checkpoint-publication interruption, resumed exact state,
immutable collisions, rollback of provisional notebook rows and completed-run
resume without training. Tests and reporting consume the phase compute budget.

### Phase 7B: Readiness, Common Ridge And Timing

Use two independently initialized development replicas with opposite rotation
signs. Follow Phase 6 initialization: Adam .001, batch 256, 15 epochs, with one
common extension to 30 if validation indicates unfinished initialization.
Use the same initialization across both excursions and every policy. Show
nominal validation curves and frozen/zero-displacement/constant-mean baselines.

Check frozen errors at both excursions and run at most one capability fit per
development replica and excursion, using 256 dedicated diagnostic-role pairs
and at most 20 epochs. These models never initialize a scientific trajectory.
Show whether a shifted deficit is recoverable. Both amplitudes stay in scope
when numerically valid, even if one shows little learning opportunity or poor
recoverability; annotate that result rather than selecting the better-looking
context. Do not widen the model, alter the likelihood variance, or invent a
third task in response to weak effects.

Run fixed-.025 development paths at $r\in\{0,.01,.1\}$ for both replicas and
excursions: at most 12 paths. Choose a **single common ridge** minimizing mean
current-law budget squared error across the two equally weighted excursions
and replicas; ties prefer lower measured learning time, then smaller ridge.
Display nominal/endpoint trade-offs for every candidate. This selection favors
a calibrated strong fixed control and does not inspect adaptive outcomes.

Time padded initialization, whole trajectories, current-law/endpoint inference,
score/Jacobian work, solver, observer, checkpoint I/O and notebook updates.
Measure actual storage and peak memory. Audit at most four representative
frozen development solves at doubled optimizer budget; these check numerical
adequacy at unchanged pi, not an oracle-pi grid. Apply necessary solver repairs
to every arm before selection. Reforecast 7C/7D from measured costs; a doubling
of input pixels is not assumed to equal a doubling of wall time.

### Phase 7C: Bounded Observer Calibration And Frozen Design

On the common-ridge fixed development paths, run shared-score shadow filters
with $\eta\in\{10^{-5},10^{-4},10^{-3}\}$ and the inherited integrated-
acceleration $Q$ model. Pool predictive innovation log scores with equal weight
for each excursion, replica and segment. Display both calibration and mean-
motion/uncertainty terms; score predictability is not proof of parameter-truth
accuracy. Retain the two highest-scoring finite settings for on-policy study
under **each** Kalman action formula, with identical tuning opportunity.

Run those candidates across both development replicas and excursions: at most
16 candidate paths, plus four covariance-only paths. Reuse the four selected-
ridge fixed paths. Select one process scale per Kalman condition by the same
mean current-law squared-error criterion as 7B, with exact ties preferring
lower learning time, then smaller eta. Keep one setting per policy across both
amplitudes, lengths and stationary controls. Run the selected four conditions
on two development stationary replicas and publish false-adaptation behavior;
do not repeatedly retune against this check.

Allow at most one adjacent decade extension of the shared eta grid if a
specific numerical, persistent-innovation or boundary-selection diagnosis
warrants it. Evaluate the additional shadow candidate for both formulas and
allow at most one added on-policy candidate per formula, eight extra paths
total. Omit this optional extension first if measured resources are tight.
Do not choose an extension from confirmation outcomes or tune arbitrary
per-coordinate gains, upper action limits or condition-specific ridges.

Audit the fixed-anchor score map without fitting population optima. At saved
initial, mid-ramp and turning states, use at most 128 already-seen training
examples per audit to compare observed scores evaluated at the common anchor
and at the learner, with their respective diagonal mappings. Under the local
map, their difference should approximate the known parameter displacement.
Report the residual relative to sampling variability and the observer-noise
scale. These reused examples are diagnostics only: do not assimilate them,
advance q, or call them independent fresh evidence. This is a local consistency
check, not proof that the inferred latent level equals $\theta^\star$.
Inspect nested already-seen noise calibration, innovations, lag and covariance
repairs. Poor consistency is an important outcome; do not silently reanchor
or replace the primary controller to conceal it.

Before any fresh outcome is exposed, seal the renderer, likelihood, schedules,
common ridge, initialization/solver rules, both eta selections, all candidates
and reasons, source identities, fresh seeds/cohort, metric definitions and
resource forecast. Preserve every development failure and unfavorable result.
Proceed with the best finite supported settings without an efficacy gate;
correctness failures require repair/versioning rather than interpretation.

### Phase 7D: Fresh Paired Exploration

Target eight fresh changing replicas **per excursion**, using the same eight
replica identities across amplitudes, and four fresh stationary replicas.
Balance each changing cohort across the four sign/length combinations with
two replicas per combination. Stationary replicas cover the four combinations
once, retaining a sign only for their endpoint panels. Four policies yield
**64 changing plus 16 stationary trajectories, 80 total**. A stationary path
is not duplicated for each endpoint amplitude.

Interleave complete four-policy blocks across excursions and stationary
controls so intermediate inspection contains progress on all contexts. The
eight paired amplitude identities are the replication units for any amplitude
interaction; do not treat their 16 amplitude-specific blocks as independent.
Use separately named stationary replica draws. Four-policy blocks
must be complete before entering paired inference. Partial blocks are visible
progress and resume to the same allocation.

Freeze the count using remaining budget, measured cost plus 25% contingency,
and at least one reserved active hour for closeout. If needed, reduce equally
to four changing replicas per amplitude while keeping four stationary replicas:
**48 total paths**, balanced across sign/length. Do not drop the harder amplitude,
weaken fixed .025, remove a condition, or shorten a regime to obtain significance.
If even the reduced cohort cannot fit, report the resource limit and completed
development rather than exceed the allowance. The frozen cohort is not expanded
because interim intervals look promising or nearly significant.

Report all three adaptive-versus-fixed contrasts, plus discrepancy-versus-
velocity and discrepancy-versus-covariance, separately by excursion and for
the stationary cohort. Use 2,000 named whole-replica paired bootstrap draws,
shared across outcomes and preserving paired amplitudes. Stratify by sign and
length when there are at least two replicas per stratum; otherwise use the
ordinary paired-replica bootstrap with the limitation stated. Label intervals
nominal 95%, conditional on development-selected settings and exploratory
multiple contrasts. No checkpoint or landmark pseudoreplication, equivalence
claim from nonsignificance, or automatic multiplicity-driven enlargement.

### Phase 7E: Concise Reporting And Interpretation

Append **Phase 7: Tracking Discrepancy During Sustained Drift** after the Phase 6
conclusion in `mnist_experiment/plan17_results.ipynb`. Open with the motivation,
the distinction between population velocity and learner discrepancy, and a
plain-language definition: **current-law budget RMS measures typical landmark
error while following the changing stream; lower is better**. State explicitly
that these are independent post-batch holdouts, not training losses.

Keep the main addition to four multi-panel figures and two compact tables:

1. Padded examples, both angle laws and schedules, all on the incoming-pair
   clock; a small frozen/capability readiness summary.
2. Development tracking-versus-retention results with selected settings and
   a concise observer/anchor-consistency diagnostic.
3. Fresh current-law RMS and within-one-pixel accuracy for both excursions
   and stationary controls; a compact aligned nominal/endpoint retention view.
4. Aligned applied weights and discrepancy/velocity/uncertainty summaries,
   with paired tracking-effect intervals. Link detailed per-coordinate and
   per-regime diagnostics rather than printing them inline.

Use one settings table and one conclusion table with fresh block counts,
current-law RMS gaps and intervals, nominal/endpoint trade-offs and compute
ratios. Label every curve and frozen reference. Keep outcome scales comparable
across conditions, use one consistent incoming-pair axis, and show exact regime
boundaries for each length. Do not interpolate different-length trajectories
into extra observations or obscure their timing by averaging unmatched grids.
Summaries distinguish a useful conditional policy, a retention trade-off,
unresolved uncertainty, poor observer calibration or a task with little
adaptation opportunity. No hypothesis is declared proved from attractive pi
curves or posterior uncertainty alone.

Near the top of the notebook, show current phase/revision, completed/planned
paired blocks by amplitude/stationarity, active accepted checkpoint, heartbeat,
charged compute, remaining-time forecast and storage. Publish atomically at
stage/block boundaries, graceful stops and at most every 15 active minutes.
Historical notebook cells and sealed report generations remain intact. Pending
or provisional evidence must be conspicuous and excluded from completed-block
inference. The notebook loads artifacts only; it never trains, downloads,
repairs artifacts or computes large Fisher matrices.

### Phase 7 Resources, Autonomy And Recovery

Execution authorization on 2026-11-05 grants a **new maximum of 20 cumulative
active wall-clock compute hours**. Preserve approximately 14.318 hours for Phases
0--5 and 4.30 hours for Phase 6 separately. No old allowance is reset or
transferred. Account for acceptance, pilots, discarded revisions, retries,
evaluation, report generation, audits and I/O as well as learning.

| Sub-phase | Provisional active compute | Deliverable |
| --- | ---: | --- |
| 7A: implementation acceptance | 0.5--1 h | Padded renderer, conditional policy, tested recovery |
| 7B: readiness, ridge and timing | 0.75--1.5 h | Both contexts, common ridge, measured resource forecast |
| 7C: observer calibration | 1.5--3.5 h | Bounded diagnostics and frozen recipes/cohort |
| 7D: fresh exploration | 4--8 h | Target 80 paired-condition paths; resource fallback 48 |
| 7E: reporting and closeout | 0.5--1 h | Concise conclusions and integrity audit |
| **Subtotal forecast** | **7.25--15 h** | Approximately **9--19 h** with contingency |
| **Authorized hard allowance** | **20 h** | All active computation |

Initial implementation is provisionally **4--8 engineering hours**, separate
from active experiment compute; computational tests are included in 7A. Phase
6's measured 4.30-hour complete study is the timing reference, but this phase
adds padded inputs, current-law inference, two excursions and more fresh paths.
Do not treat the forecast as measured until 7B. All upper bounds plus optional
expansions cannot be spent independently; the remaining-time calculation governs.

Retain the shared **500 GB peak-artifact ceiling**, **20 GB actual free-disk
reserve**, and an under-10-GB incremental target. Store source IDs, seeds and
compact transform plans instead of all rendered panels. Bounded recovery
checkpoints, scalar metrics, selected audits and final states suffice; no
per-arrival archive of full models or dense matrices is needed. Charge actual
GPU/CPU resources and do not infer compute ratios from parameter count alone.

Require `--resume`, bounded invocations, status/stop commands, one family writer
lock and persistent allocation accounting. Atomically checkpoint learner/Fisher/q,
both filter means/covariances, anchors/noise state, RNGs, pending render indices,
accepted metric offsets, evaluation progress and charged time. Separate a
durably accepted batch from a provisional report row. Resume preserves every
completed byte, never assimilates an observation twice, and never renews the
compute allowance. Make validation/reporting resumable or bounded and check
stop requests inside their loops, targeting a graceful stop within one minute.
Verify abrupt process death during learning and during publication, not only
a polite stop between trajectories.

After authorization, Sol may implement numerical/software corrections and the
bounded development decisions above without waiting for a model switch or user
review. Record materially significant mathematical changes in the notebook and
handoff, preserve their prior artifacts, and use new source/config identities;
do not pool incompatible revisions as one confirmation cohort. Resume existing
scientific versions only when their manifests match. Scope expansion, a new
controller family, a new task, or additional compute needs a new agreement.

Maintain executable entry points and a verified restart command in
`PLAN17_HANDOFF.md` after implementation. Local execution needs no tmux; remote
execution can use it when requested. Conclude with final notebook publication,
sealed block/source checks, completed-resume verification, resource totals and
no queued follow-on experiment.

### Phase 7A Implementation And Acceptance Record

Implementation on 2026-11-05 is in `kalman_pi/discrepancy/`, with separate
source identities, resolved configurations and a `plan17-phase7` allocation.
The conditional action receives the actual learner vector, predicts level
uncertainty with its level/velocity cross covariance, and omits the marginal
old q term. The velocity action matches the inherited recommendation in focused
tests. Current-law holdouts use the latest accepted batch's law; repeated panels
share inference by immutable panel hash. Frozen-current predictions are cached
once per replica/context and remain a changing reference curve.

The first CUDA acceptance exposed PyTorch's unsupported deterministic adaptive-
pooling backward. Fixed 10-by-10 average pooling on the 20-by-20 feature map
implements the same 2-by-2 averaging, with value/gradient agreement within
float64 rounding, and deterministic CUDA execution remains enabled. The
prototype CPU v1 root (69 units) and failed CUDA v1 root are retained; their
source identities are not current resume targets. No production learning had
started. The corrected CUDA v2 root completed all 69 units / 36 tiny paths after
an intentional six-unit stop; all six earlier completion seals and the frozen
scientific manifest remained unchanged.

Independent review also moved the four doubled-budget solver checks to the
selected common ridge, covering nominal, mid-ramp and turning states across
both excursions. The observed-score diagonal map remains an approximation:
off-diagonal Fisher structure can yield apparent discrepancy even when the
learner is on target. Keep the anchor-consistency audit visible rather than
interpreting Kalman covariance as proof of calibrated tracking.

An overlapping validation reservation initially caused an apparent decrease
in reported active time when it settled. The acceptance wrapper now holds the
family writer lock until settlement; tests verify that a second writer cannot
enter during that interval. A documented 60-second conservative entry covers
unrooted preflight, agent checks and manifest review. No charged history or
scientific unit was erased. CPU recovery tests include process death during a
trajectory and interruption before checkpoint publication, with provisional
report rollback and exact continuation. Acceptance evidence is excluded from
applied inference.

Final acceptance passed **68 unit tests and six integration tests** in 140.16
seconds. CUDA v2's 69 completed units passed full-file/source validation; all
six pre-stop seals were unchanged. The Phase 6 read-only audit again verified
all 113 completed units and every scientific source hash. Production uses the
separately versioned root `discrepancy_v1` after these checks.

### Phase 7 Development Freeze Record

All 40 development trajectories completed. The immutable selection was sealed
at **2026-11-05 18:03:08 UTC**, before fresh outcomes: common ridge **0**,
velocity process scale **$10^{-4}$**, discrepancy process scale **$10^{-5}$**,
eight shared changing replica identities at each of 25/45 degrees, and four
separate stationary identities. The four-policy allocation is **80 paths /
20 paired blocks**. The endpoint amplitudes remain paired, not independent
new replicas.

The fixed-control screen chose ridge zero using current-law budget squared
error. Innovation screening retained $10^{-5}$ and $10^{-4}$ for both action
formulas; on-policy development selected their separate settings. No optional
decade extension was used: the conspicuous limitation is fixed-anchor map
consistency, not a demonstrated numerical filter defect that another process
scale would resolve. Discrepancy candidates and stationary checks were
unfavorable; all are retained, and confirmation proceeds without an efficacy
gate or a changed action limit.

Measured median development online wall time was **248.735 seconds per path**
and initialization time **121.881 seconds per replica**. The sealed fresh
forecast, including its path contingency and initialization allowance, is
**7.916 additional active hours**, with at least one further active hour
reserved for closeout. Approximately 2.97 Phase 7 hours had been charged at
this transition; the 20-hour allowance and historical ledgers are unchanged.
These are forecasts, not final resource use. Fresh outcomes cannot change the
frozen cohort, policies, common ridge or process scales.

### Phase 7 Results And Closeout

All **80 fresh trajectories / 20 paired blocks** completed under the frozen
design: eight changing replicas per excursion, paired across excursions, and
four separate stationary replicas. No significance stopping, post-confirmation
tuning or optional cohort expansion occurred. The chronological Phase 7
section and compact conclusion are in `plan17_results.ipynb`.

The **existing velocity-based Kalman policy has a modest positive result** in
this context. The new conditional-discrepancy implementation does not. Here
are policy-minus-fixed gaps in current-law budget RMS; negative is better:

| Context | Velocity gap, pixels [nominal 95% CI] | Discrepancy gap, pixels [nominal 95% CI] |
| --- | --- | --- |
| 25 degrees, 8 replicas | -0.01425 [-0.01777, -0.01107] | +0.09290 [0.07893, 0.10710] |
| 45 degrees, 8 replicas | -0.01307 [-0.01803, -0.00810] | +0.07995 [0.06505, 0.09577] |
| Stationary, 4 replicas | -0.01394 [-0.02341, -0.00717] | +0.07888 [0.05464, 0.10270] |

Fixed current-law budget RMS averaged 0.94679 / 1.00133 pixels at 25/45 degrees.
Velocity's gains occur in **both outward and return ramps**, not just quiet
periods: all four corresponding nominal intervals favor velocity. Its whole-
path within-one-pixel accuracy gains are **+0.814 pp [0.547, 1.088]** and
**+0.793 pp [0.415, 1.170]**, respectively. Current-law RMS favored velocity
in 8/8 and 7/8 replicas. Covariance-only tracking is unresolved at 25 degrees
and worse at 45 degrees; suppressing movement sensitivity is not uniformly
successful.

This is not a joint retention/acquisition win. Velocity's nominal-domain gap
is unresolved at 25 degrees and favorable at 45 degrees, but its fixed endpoint
panel gaps are **+0.02247** and **+0.00831** pixels, with both nominal intervals
above zero. Those panels cover the entire path, not just the endpoint regime.
Retention was not matched. Stationary endpoint panels measure untrained-domain
generalization. All secondary intervals and contrasts remain in the artifacts.

Velocity averages adaptation weights about .00865 on changing paths and .01023
on stationary paths, with no clipping or fallbacks. Discrepancy averages .19265
and .19013, with upper-bound frequencies averaging **83.0% / 73.7%** within
trajectories; neither family used a nonfinite fallback. Poor fixed-anchor
consistency and aggressive stationary actions support concern about the level
proxy. They do not uniquely identify the cause or refute the conditional
quadratic-risk identity.

At equal incoming-data budgets, velocity's learning / online-wall ratios are
**.817 / .950** at 25 degrees and **.811 / .951** at 45 degrees. Discrepancy is
about **2.20 times** the learning cost and **1.49--1.51 times** online wall time.
These are measured policy-package costs, not parameter-count estimates or
compute-matched comparisons. The findings do not establish an exact LAN regime,
globally optimal pi, superiority to every constant weight, or sample savings.
The intervals are exploratory and conditional on development selection; the
small stationary cohort especially limits interpretation.

Final acceptance passed **75 unit and six integration tests (81 total)** in
154.64 seconds, including actual interruption and exact resumed state. The
reporting-only closeout shortened the conclusion table, retained every detailed
contrast, removed stationary drift shading, fixed clipped labels, and separated
signal scales without changing scientific sources or data. Earlier wording and
plot-test harness failures are retained in the charged acceptance ledger.

The final audit verified **207 completed units, 42 scientific sources, all
80 fresh paths and 20 paired blocks**, plus all **52 inherited notebook cells
and metadata**. The final notebook has 65 cells and 19 embedded nonblank figures.
All 49 early production seals remained unchanged through learning; an actual
completed `--resume` returned `already_completed` and preserved all 207 seals
and the final notebook bytes. Control status confirms no active writer.

Final report: `discrepancy_v1/report/generation-c77d0410aa477483de140421`.
Final audit: `discrepancy_v1/closeout_validation-08fd332a337f.json`, referenced
by `closeout_latest.json`. Charged Phase 7 compute is **9.336 hours**, including
tests, retained failures, publication, resume validation and conservative
read-only closeout overhead. The earlier **18.619 hours** remain separately
preserved. Production occupies about **2.205 GB**; shared cache use is about
**171.1 / 500 GB**, with approximately **577.6 GB** free. The notebook progress
clock is the last scientific snapshot; the allocation ledger also includes
subsequent closeout charges. No further experiment is queued.

### Phase 7 Practical Interpretation

Velocity Kalman is a promising adaptation-weight selector for a single
continuing stream when repeated tuning trials are unavailable. Once configured,
it uses past score observations to choose the next weight, without alternative
learner runs or online holdout feedback. The modest tracking gains here, with
slightly lower measured online wall time, motivate this application. Choosing
a useful average weight could itself be valuable, even if adaptive timing
explains only part of the gain. Observer settings were selected on development
trajectories; their transfer to unfamiliar streams remains an open empirical
question. Fixed endpoint prediction worsened, so both tracking and retention
remain relevant. This interpretation is also the notebook's Phase 7 conclusion.

A post-completion artifact review used every accepted batch in the eight fresh
velocity trajectories per amplitude. The following are within-trajectory means,
then equally weighted replica means; the initial hold includes the 16 cold-start
batches at .025. Different trajectory lengths use their recorded regime bounds.

| Interval | Mean pi, 25 degrees | Mean pi, 45 degrees |
| --- | --- | --- |
| Initial hold | .010803 | .010803 |
| Outward ramp | .008708 | .008325 |
| Return ramp | .006248 | .004707 |
| Final hold | .009574 | .009122 |
| Whole trajectory | .008932 | .008366 |

The turnaround neighborhood (last quarter of the outward ramp and first quarter
of the return) has mean weights .005962 / .004517. Relative to each trajectory's
last quarter of the initial hold, the mean within-trajectory ratio of movement
energy to noise, $\rho_t=mS_t/V_t$, falls to .489 / .253 of its earlier value.
The comparison includes the velocity-uncertainty contribution to $S_t$. Absolute
motion and uncertainty energies can grow while this ratio falls: noise grows
faster. The nominal old-summary concentration $q_t$ also decreases, reinforcing
the action reduction through $\pi_t=(\rho_t+mq_t)/(1+\rho_t+mq_t)$ outside cold
start and clipping. No velocity actions clipped or fell back. These quantities
are reconstructed from sealed `steps.json` records and the specified q recursion;
they are descriptive, not fresh hypothesis tests.

The smooth rotation law has zero angular velocity at maximum excursion, so its
turnaround does not require maximum population velocity. The wider dip also
reflects estimated noise and accumulated-weight concentration. This explains
the controller's arithmetic, not whether its fixed-anchor map measures true
population motion accurately or whether the timing itself causes the benefit.
No new trajectories or scientific settings were introduced for this review.
