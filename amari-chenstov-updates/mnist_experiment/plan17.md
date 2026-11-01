# Implementation Plan 17: Observation-Clock Adaptive Pi

**Motivation:** obtain useful continual learning per scarce observation and per
unit of compute, while retaining previously learned recognition. A fixed
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

**Status:** draft for review, 2026-11-01. No execution or results are implied.
**Roles:** Astra supplies this design; Sol implements and executes all phases
after authorization, with authority to correct defects and keep useful work
running without further model switches or recurring user approvals.
**Compute:** **20 cumulative active wall-clock hours for Plan 17**, separately
authorized from Plan 16. This is not permission to run both consecutively.
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

Proposed interfaces, **to be implemented and tested in Phase 0**:

```bash
~/.venv/bin/python -u -m mnist_experiment.kalman_pi.run \
  --config mnist_experiment/kalman_pi/configs/default.json \
  --root cache/mnist_experiment/plan17/production_v1 \
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
