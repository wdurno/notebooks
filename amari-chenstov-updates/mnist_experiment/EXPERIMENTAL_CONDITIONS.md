# Phase 9 experimental conditions

This guide defines the initial five-replica controller screen and its LFU
isolation extension. Controller-screen names describe the one setting changed
from the center condition.

## Shared mechanics

The environment coordinate $p=\mathbb P(M_i=1)$ is the proportion of digit 9
observations. Each replica traverses 100 evenly spaced values from $p=0$ to
$p=1$, using 128 new observations per step. Conditions within a replica share
their initialization, observations, and reference path.

The adaptation weight $\pi_t$ is different from $p$. It controls both:

- the EWC old-to-new odds, $\lambda_t=(1-\pi_t)/\pi_t$;
- how much the auxiliary Fisher estimate trusts the current fresh estimate.

Every condition uses the same rank-8-plus-diagonal Fisher tracker. For steps
after initialization, its general prediction is

$$
\widehat{\mathcal I}^{\mathrm{pred}}_t
=\widehat{\mathcal I}_{t-1}
+\widehat{(C+R):u_{t-1}},
$$

then combines the LFU prediction with the current batch Fisher:

$$
\widehat{\mathcal I}_t
=(1-\pi_t)\widehat{\mathcal I}^{\mathrm{pred}}_t
+\pi_t\widehat{\mathcal I}^{\mathrm{fresh}}_t.
$$

The candidate is symmetrized and reduced to a PSD rank-8-plus-diagonal
representation. Optimization uses a budget of 50 L-BFGS iterations per step.
Every initial controller-screen condition uses the full correction
$\widehat{(C+R):u}$. The LFU-isolation conditions vary that correction while
holding the remaining mechanics fixed.

## Oracle role

Conditions share an independently fitted reference-optimum path
$\theta^{\mathrm{ref}}(p)$, but they do **not** follow it. Starting from the
same paired initialization and observation stream, each condition's EWC
optimizations generate its own closed-loop parameter path. Those paths may and
do diverge.

The reference path supplies oracle displacements for trend-error diagnostics.
A high-sample reference Fisher is also evaluated at each condition's actual
current parameter to measure Fisher-tracking error. Neither quantity chooses
$\pi_t$, changes the LFU estimate, or enters the EWC objective for the
`fixed_unified` and `optimal_plugin` policies used in Phase 9. The separately
defined `optimal_oracle` policy would use oracle displacement information, but
it is not one of these production conditions.

There is one genuine source of oracle-quality assistance: every condition
starts with the same high-quality Fisher estimate at $p=0$. A later deployment
comparison should remove reference-path and high-sample-Fisher diagnostics and
initialize the Fisher from a realistic online compute budget. Removing the
diagnostics should not change a fixed or plug-in trajectory; that invariance is
itself worth testing.

## Two half-lives

The experiment contains two distinct exponential-memory settings:

- **Controller trend half-life** is measured in environmental $p$-distance.
  It smooths the estimated unregularized displacement trend $u_t/\pi_t$ and
  residual moments used by the plug-in controller. It also sets the duration
  of the controller's cold start. With 100 $p$ points, half-lives of $0.10$,
  $0.20$, and $0.40$ are approximately 10, 20, and 40 steps.
- **LFU ridge half-life** is measured directly in steps. It smooths noisy
  estimates of $(C+R):u$ along a locally coherent direction. It is fixed at
  eight steps in every Phase 9 condition.

### Controller trend recursion

Let $\widehat d_t$ be the predictable estimate of the unregularized local
displacement, $q_t=N_{\mathrm{eff},t}^{-1}$, $m$ the current batch size, and
$\widehat T_t$ the estimated covariance trace. Before observing the step-$t$
displacement, the plug-in controller calculates

$$
S_t=\|\widehat d_t\|^2,
\qquad
\pi_t^{\mathrm{plugin}}
=\frac{S_t+\widehat T_tq_t}
{S_t+\widehat T_tq_t+\widehat T_t/m+\varepsilon},
$$

then clips this value to $[\pi_{\min},\pi_{\max}]$. During the first half-life
of environmental movement, it instead uses the cold-start composition

$$
\pi_t^{\mathrm{cold}}
=\frac{m}{N_{\mathrm{eff},t}+m}.
$$

Only after the resulting EWC step accepts displacement $u_t$ is the state
updated. For controller half-life $h$ and environmental increment $\Delta p_t$,

$$
g_t=1-2^{-\Delta p_t/h},
\qquad
\widehat d_{t+1}
=(1-g_t)\widehat d_t+g_t\frac{u_t}{\pi_t}.
$$

The same gain updates the covariance-trace moments:

$$
r_t=u_t-\pi_t\widehat d_t,
\qquad
a_t=\pi_t^2(q_t+m^{-1}),
$$

$$
R_{t+1}=(1-g_t)R_t+g_t\|r_t\|^2,
\qquad
A_{t+1}=(1-g_t)A_t+g_ta_t,
\qquad
\widehat T_{t+1}=\frac{R_{t+1}}{A_{t+1}+\varepsilon},
$$

while effective size evolves as

$$
q_{t+1}=(1-\pi_t)^2q_t+\frac{\pi_t^2}{m}.
$$

Consequently, an old trend contribution is halved after the environment moves
by $h$ in cumulative $p$-distance. Smaller $h$ reacts faster but is noisier;
larger $h$ smooths longer but lags curvature. In the current controller, $h$
also sets the cold-start duration. Therefore, a `trend-h-*` comparison jointly
changes EMA responsiveness and cold-start length; it does not identify which
of those two effects caused a performance difference.

Therefore, `trend-h-010` changes the controller's memory and cold-start length;
it does **not** change the LFU ridge regression half-life.

## Initial controller conditions

| Condition | $\pi_t$ rule | Difference from center |
|---|---|---|
| `control` | Fixed $\pi_t=0.05$ | Non-adaptive paired baseline; full LFUs still run. |
| `center` | Plug-in controller, clipped to $[0.05,0.95]$ | Controller trend half-life $0.20$; reference condition. |
| `pi-min-001` | Center rule clipped to $[0.01,0.95]$ | Allows stronger retention of old information. |
| `pi-min-010` | Center rule clipped to $[0.10,0.95]$ | Requires at least 10% new-information weight. |
| `pi-max-080` | Center rule clipped to $[0.05,0.80]$ | Limits the maximum new-information weight. |
| `pi-max-100` | Center rule clipped to $[0.05,1.00]$ | Allows the EWC weight to fall to zero. |
| `trend-h-010` | Center bounds | Controller trend half-life and cold start become $0.10$ in $p$-distance. |
| `trend-h-040` | Center bounds | Controller trend half-life and cold start become $0.40$ in $p$-distance. |

For the plug-in conditions, the raw $\pi_t$ minimizes the estimated one-step
Euclidean risk using the lagged displacement trend, estimated covariance trace,
effective old sample size, and current batch size. Clipping then enforces the
condition's $[\pi_{\min},\pi_{\max}]$ interval. Smaller $\pi_t$ retains more old
information; larger $\pi_t$ adapts more strongly to the current batch.

## LFU-isolation conditions

The isolation screen uses three Fisher-update methods:

| Method | Applied correction $\widehat\Delta_t$ |
|---|---|
| No LFU (`ema`) | $0$ |
| AC-only | $\widehat{C:u_{t-1}}$ |
| Full LFU | $\widehat{(C+R):u_{t-1}}$ |

Here `ema` names the no-LFU Fisher blend; it is unrelated to the controller
trend EMA. All three conditions still calculate per-batch $C$ and $R$ for
paired diagnostics. AC-only and full LFU use the same eight-step directional
ridge. No-LFU skips only the correction in the Fisher prediction:

$$
\widehat{\mathcal I}_t
=(1-\pi_t)\widehat{\mathcal I}_{t-1}
+\pi_t\widehat{\mathcal I}^{\mathrm{fresh}}_t.
$$

| New condition | Controller | Applied correction | Named comparison |
|---|---|---|---|
| `fixed-005-no-lfu` | Fixed $\pi=.05$ | None | Existing fixed-.05/full-LFU control |
| `fixed-010-no-lfu` | Fixed $\pi=.10$ | None | Reference for the two fixed-.10 corrections |
| `fixed-010-ac-only` | Fixed $\pi=.10$ | $C:u$ | Fixed-.10/no-LFU |
| `fixed-010-full-lfu` | Fixed $\pi=.10$ | $(C+R):u$ | Fixed-.10/no-LFU |
| `adaptive-h010-no-lfu` | Plug-in, controller $h=.10$ | None | Existing adaptive-$h=.10$/full-LFU condition |
| `adaptive-h010-ac-only` | Plug-in, controller $h=.10$ | $C:u$ | Existing adaptive-$h=.10$/full-LFU condition |

All comparisons are paired within replica. Existing full-LFU dependencies are
reused as immutable controls and are not recomputed. Reported effects are
always the named condition minus its named comparison; consult the table when
interpreting the sign.

## Adaptation screen

This compact follow-up holds the no-LFU Fisher update fixed and varies the
controller. Every new condition uses the completed adaptive $h=.10$ no-LFU
condition as its paired comparison.

| New condition | Controller | Purpose |
|---|---|---|
| `adaptive-h020-no-lfu` | Plug-in, $h=.20$ | Intermediate trend memory |
| `adaptive-h040-no-lfu` | Plug-in, $h=.40$ | Long trend memory |
| `fixed-100-no-ewc` | Fixed $\pi=1$ | Current-batch likelihood with zero EWC penalty |

For `fixed-100-no-ewc`, $(1-\pi)/\pi=0$, so the EWC quadratic vanishes. The
current experiment runner still calculates Fisher diagnostics; its predictive
trajectory is a valid no-EWC comparison, but its runtime is not an optimized
no-EWC deployment benchmark. After initialization, its auxiliary recursion also
has no memory: it reports the instantaneous empirical Fisher $Z_t$ from the
current batch. This is a diagnostic of a different process, not another name
for the current-batch learner.

## Plan 2 low-data regime discovery

Plan 2 searches for the online data budget at which compressed historical
information becomes useful. It is preparatory: LFUs and replay comparisons are
deferred until this screen finds a regime where current-batch-only learning is
not adequate.

The initial screen holds the 100-point linear path from $p=0$ to $p=1$ fixed
and varies

$$
m\in\{1,2,4,8,16,32,64\}.
$$

The completed $m=128$ adaptation screen remains the easy-regime anchor. Each
lower-$m$ stream is planned as a deterministic prefix of the same ordered
128-observation master batches within a replica. Thus, conditions share the
exact $p=0$ model, initial Fisher, partitions, and observations, while larger
$m$ values add observations to smaller ones.

### Initial conditions

All three conditions use rank 8 plus a diagonal, an L-BFGS budget of 50, and
the no-LFU Fisher recursion. The screen therefore asks whether old information
and the controller help before reintroducing derivative corrections.

| Condition | Original learning process | Auxiliary Fisher process | Role |
|---|---|---|---|
| `no-ewc-pi100` | Current batch only; fixed $\pi=1$ makes the EWC penalty zero | Instantaneous empirical Fisher $Z_t$ after initialization | Primary control |
| `fixed-ewc-pi010` | Current batch plus EWC-compressed history at fixed $\pi=.10$ | Recursive Fisher summary | EWC mechanism control |
| `adaptive-ewc-h010` | Current batch plus EWC-compressed history at plug-in $\pi_t\in[.05,.95]$ | Recursive Fisher summary | Applied adaptive treatment |

The two entries in each row describe coupled but distinct processes. In
particular, the rank of the no-EWC condition's instantaneous empirical Fisher
does not describe the rank of information learned by its optimizer.

The fixed weight and half-life are starting values, not claimed optima. If
fixed EWC succeeds while adaptive EWC fails, the experiment has found a
controller problem rather than disproved the compressed summary.

### Primary predictive metrics

Plan 2 uses four primary metric trajectories, evaluated separately within each
environmental mixture value $p_t$:

1. digit-9 one-vs-rest accuracy $A_{9,\mathrm{OvR}}(p_t)$;
2. digit-9 precision at prevalence $p_t$;
3. digit-9 recall; and
4. environmental multiclass accuracy.

Let $r_9$ be recall on true nines, $f_9$ the rate at which non-nines are
predicted as 9, $s_9=1-f_9$, and $A_{\mathrm{non9}}$ exact multiclass accuracy
conditioned on a non-nine. The environmental metrics are

$$
A_{9,\mathrm{OvR}}(p_t)=p_t r_9+(1-p_t)s_9,
$$

$$
P_9(p_t)=\frac{p_t r_9}{p_t r_9+(1-p_t)f_9},
$$

and

$$
A_{\mathrm{env}}(p_t)=p_t r_9+(1-p_t)A_{\mathrm{non9}}.
$$

Precision is undefined when its denominator is zero and is reported as
missing, not fabricated. The fixed holdout estimates the class-conditional
rates; its empirical digit prevalence is not substituted for $p_t$.

Replicas are the statistical units. Each metric is calculated per replica and
$p_t$, then averaged across replicas at that same $p_t$ with pointwise
variability retained. The legacy field `nine_accuracy` means 9 recall. Balanced
accuracy is not a Plan 2 target metric. NLL, calibration, old-digit retention,
Fisher diagnostics, and compute remain supporting evidence.

### Decision gate

A target-regime candidate must satisfy both conditions:

1. `no-ewc-pi100` no longer learns digit 9 adequately or stably at the
   available observation budget.
2. At least one EWC condition obtains a practical acquisition advantage
   without unacceptable old-digit loss.

Poor no-EWC NLL is not enough when its digit-9 classifier still works. Likewise,
high 9 recall is not enough when precision or 9 OvR accuracy reveals excessive
false positives. A budget where every learner fails is also not useful.
Selection uses expected trajectories, paired effects, uncertainty, and
fixed-exposure comparisons rather than one replica or one threshold crossing.

If no-EWC still works at `m=1`, the next intervention is trajectory geometry or
implicit rehearsal, not a still smaller batch.

### Exposure and calibration

New scalar artifacts record the actual batch class counts, repeated and unique
observation counts, and cumulative exposure before and after each model update.
The terminal batch is available to the auxiliary Fisher diagnostic but is not
charged as optimizer-consumed data because no subsequent model update occurs.

Digit-9 acquisition is summarized from the mean post-update trajectories
across replicas at each $p_t$. A recall-only 90% crossing is not a target-regime
gate because it can be attained by predicting 9 too often. Any later durable
criterion must include a false-positive-sensitive metric. If it is not met,
report all four primary metrics at common fixed observation budgets.

### Equivalent Data Multiplier

The Equivalent Data Multiplier (EDM) is the no-EWC exposure required to match
an EWC condition at the same $p_t$, divided by the EWC exposure. Its primary
denominator is cumulative unique digit-9 observations. Report each primary
metric separately and a joint EDM that requires all four simultaneously.

EDM uses the tested no-EWC data-budget envelope. Sparse results are brackets,
not interpolated point claims: $(2,4]$ means twice the data failed to match the
EWC outcome and four times succeeded. An unmatched largest budget is reported
as right-censored. Data, optimizer work, elapsed compute, and storage remain
separate efficiency measures.

The coarse screen selected $m=8$ provisionally. Phase 3 adds replicas 4 and 5
at $m\in\{8,16,32\}$, retaining the same three conditions, 100-point path,
rank-8-plus-diagonal summary, no-LFU recursion, and L-BFGS budget 50. The three
budgets represent the target, its upper neighbor, and the tested no-EWC
equivalence boundary.

All five replicas support retaining $m=8$ as the low-data target. At the last
state below $p=.5$, adaptive EWC had the best mean joint and environmental
performance, but it was not decisively separated from fixed EWC. The strict
joint EDM is bracketed by no-EWC $m=16$ and $m=32$; paired resampling makes the
lower budget unresolved and the upper budget a credible match. Treat this as
an exposure interval, not a precise threefold estimate.

### Phase 4 controller recalibration

Phase 4 holds $m=8$, the 100-point path, rank-8-plus-diagonal Fisher summary,
no-LFU recursion, and optimizer budget 50 fixed across five paired replicas.
It screens controller half-life $h\in\{.05,.10,.20,.40\}$, then screens
$\pi_{\min}\in\{.01,.05,.10\}$ at $h=.20$. An explicit fixed-$\pi=.05$
condition separates the selected weight from the plug-in rule that proposed
it.

| Condition | Meaning | Phase 4 conclusion |
|---|---|---|
| `adaptive-ewc-h005` to `adaptive-ewc-h040` | Plug-in controller with the named trend half-life and $\pi_{\min}=.05$ | $h=.20$ and $.40$ led the screen; both were almost entirely lower-bound driven. |
| `adaptive-ewc-h020-pimin001` | Same controller with lower bound $.01$ | Feedback and trace estimates became unstable; predictive quality fell. |
| `adaptive-ewc-h020-pimin010` | Same controller with lower bound $.10$ | Exactly reproduced fixed $\pi=.10$. |
| `fixed-ewc-pi005` | Fixed sample-composition/EWC weight $.05$ | Selected applied policy for confirmation. |

The explicit fixed-$.05$ policy and `adaptive-ewc-h040` match exactly at all
500 paired replica-step points. `adaptive-ewc-h020` differs at only two points
and never exceeds $.0516$. Thus Phase 4 supports the fixed $.05$ weight on this
path, but does not validate the current plug-in controller formula. Fixed
$\pi=.10$ remains a mechanism comparator and no EWC remains the primary
control.

### Phase 5 independent confirmation

Phase 5 repeats the selected $m=8$ regime on five fresh statistical units,
replicas 6 through 10. Each has an independent $p=0$ fit, initial Fisher,
ordered stream, and reference path. The four paired conditions are no EWC,
fixed $\pi=.05$, adaptive $h=.20$ with $\pi_{\min}=.05$, and fixed
$\pi=.10$; all retain rank 8 plus a diagonal, no LFU, and optimizer budget 50.

Across $p<.5$, fixed $.05$ and adaptive $h=.20$ both obtained environmental
accuracy AUC $.682$, compared with $.532$ for no EWC. Their environmental NLL,
Brier, ECE, and non-nine-accuracy AUCs improved against no EWC in every fresh
replica. Improvements in 9 OvR accuracy and precision were positive on average
but less uniform, while recall was approximately unchanged.

The adaptive and fixed-$.05$ trajectories were exactly equal at all 500 fresh
replica-step points because adaptive $\pi_t$ remained at its $.05$ lower bound.
Accordingly, both conditions proceed to later work, but with different claims:
fixed $.05$ is the confirmed practical policy; adaptive $\pi$ remains a
diagnostic and a candidate for a future variable-speed or variable-curvature
path.

NLL remains the primary proper scoring rule. New-schema runs also record the
multiclass Brier score and 15-bin, equal-width maximum-probability expected
calibration error during the existing holdout pass; logits are not stored.

### Compute labels

Measured quantities include process wall time, per-operation wall time,
optimizer iterations and function evaluations, score-gradient and HVP counts,
peak allocated CUDA memory, peak process RSS, and artifact bytes. Peak memory
is a high-water mark, not memory-time consumption.

The Plan 2 command center reports `estimated_wall_hours`, a planning proxy based
on historical trajectory timing. The older Phase 9 command center retains its
legacy `estimated_gpu_hours` field; do not interpret that label as measured GPU
occupancy. A future accelerator-consumption claim requires CUDA-event timing or
sampled device utilization.

## Plan 3 proposed controls

Plan 3 holds the accepted `m=8`, 100-point path, rank-8-plus-diagonal summary,
and 50-iteration optimizer budget fixed. Factors are staged rather than crossed
in one grid.

| Condition family | Meaning | Role |
|---|---|---|
| Current only | Current batch, no EWC, no replay | Primary no-history control |
| EWC/no LFU | Fixed $\pi=.05$ compressed summary | Constant-memory reference |
| Bounded replay | FIFO capacities 8, 32, and 128 | Replay-capacity screen |
| Unbounded replay | Retain all online observations | Unconstrained-memory control |
| Memory-matched replay | Replay capacity derived from canonical EWC bytes | Storage-matched control |
| Hybrid | Active FIFO replay; evictions enter a disjoint EWC archive | Compression-plus-replay treatment |
| LFU variants | No LFU, AC-only diagnostic, and full LFU | Fisher-update isolation |

At each update, replay fits the current batch once together with the pre-update
buffer. The current batch enters the buffer after a successfully completed fit.
An aborted transaction inserts nothing. Pure bounded
replay discards FIFO evictions; hybrid replay compresses each eviction into the
archive exactly once. Unbounded replay never evicts and reaches at most 800
online observations here. It does not replay the original initialization
dataset, so it is not a full-data retraining oracle.

The hybrid archive has its own anchor: the parameter estimate belonging to the
archive-only EWC summary. The main learner is fitted against that anchor using
the current batch and active replay, but its fitted parameters never recenter
the archive. Afterward, only FIFO evictions are consolidated against the prior
archive, and their score Fisher is evaluated at the resulting archive anchor.
This gives every online event the disjoint lifecycle current batch, exact
replay, then compressed archive. Phase 4 includes capacities 8, 25, and 32 so
the incremental value of one replay batch can be separated from larger hybrid
windows.

The online stream samples source MNIST indices with replacement. Repeated
indices remain distinct arrival events, occupy distinct FIFO positions, and
are not deduplicated.

Plan 3 reports logical observation bytes even when MNIST indices are stored as
an implementation shortcut. Each canonical replay item costs 800 bytes: 784
uint8 pixels, an int64 label, and an int64 identity. With 24 bytes of FIFO
metadata, the 20,480-byte fixed-policy rank-8-plus-diagonal EWC summary matches
25 replay observations. Persistent learner state, peak working memory, repeated
optimizer presentations, unique exposure, wall time, and CUDA elapsed time
remain distinct quantities.

### Phase 4 history frontier

Phase 4 compared the fixed-$.05$ EWC summary, pure replay, and clean no-LFU
hybrids over five paired replicas. The table reports normalized AUC over
$0\leq p<.5$.

| Condition | Environment accuracy | 9 OvR accuracy | 9 precision | 9 recall | NLL | Persistent bytes |
|---|---:|---:|---:|---:|---:|---:|
| EWC | .682 | .838 | .587 | .555 | 1.54 | 20,480 |
| Replay B8 | .650 | .867 | .667 | .632 | 8.90 | 6,424 |
| Replay B25 | .691 | .882 | .700 | .644 | 8.48 | 20,024 |
| Replay B32 | .718 | .888 | .714 | .648 | 7.68 | 25,624 |
| Hybrid B8 | .738 | .870 | .677 | .631 | 1.44 | 26,904 |
| Hybrid B25 | .768 | .887 | .716 | .639 | 1.22 | 40,504 |
| Hybrid B32 | .774 | .890 | .723 | .639 | 1.19 | 46,104 |
| Unbounded replay | .772 | .917 | .812 | .636 | 6.11 | 633,624 |

Hybrid B32 is the bounded Phase 5 selection. It approximately matched
unbounded replay's environmental accuracy with much lower storage and better
calibration, but unbounded replay retained a clear advantage in digit-9 OvR
accuracy and precision. Hybrid B8 and B25 remain meaningful lower-memory
frontier points rather than failed conditions.

### Phase 5 LFU isolation

Phase 5 holds `m=8`, fixed $\pi=.05$, rank 8 plus diagonal, the 100-point
path, optimizer budget, stream, and initialization fixed. It reuses the
completed no-LFU EWC and Hybrid B32 trajectories and adds AC-only and full-LFU
EWC plus full-LFU Hybrid B32. The hybrid LFU is evaluated at the updated
archive anchor using the realized archive displacement and only the newly
evicted observations; active replay observations remain outside the archive.

The LFU treatments use the accepted eight-step directional ridge. Their
persistent-memory charge includes its reference direction and two dense
matrix numerators, not only the rank-8 Fisher summary. Report predictive
effects together with score-gradient/HVP counts, derivative time, correction
magnitude, direction resets, and pre-projection PSD diagnostics. Phase 5 is a
mechanism screen; Phase 6 removes scientific diagnostics before measuring
deployment cost.

The replica-6 preflight failed this gate. In the primary region, full-LFU
Hybrid B32 projected away 33.5% of candidate Frobenius norm on average and its
directional ridge reset on 97.8% of updates. EWC AC-only and full-LFU
candidates were materially indefinite on 92% and 98% of steps. These are
failed treatment diagnostics, not missing-data estimates; the remaining
replicas were deliberately not run.

### Phase 6 deployment frontier

Phase 6 reruns seven deployment-form conditions without a reference-optimum
path or online high-sample Fisher diagnostics. EWC and hybrid conditions use
rank-8-plus-diagonal Fisher EMA without LFU. Fixed conditions use $\pi=.05$;
adaptive conditions use $h=.20$, $\pi_{\min}=.05$, and $\pi_{\max}=.95$.
For adaptive hybrids, one realized $\pi_t$ weights learner EWC, archive
consolidation, and Fisher EMA.

The primary analysis is normalized AUC over $0\leq p<.5$. Fixed Hybrid B32 is
the strongest constrained-memory all-rounder: it improves environmental
accuracy and calibration over EWC or Replay B32 while using 46,104 logical
bytes. Unbounded replay uses 633,624 bytes and retains higher 9 OvR accuracy
and precision, but is tied with the hybrid on environmental-accuracy AUC.
Replay B32 is the cheapest useful learner at 11.1 mean learner seconds;
Hybrid B32 needs 48.6 seconds because archive consolidation adds a second fit.
Adaptive EWC exactly equals fixed EWC here, and adaptive hybrid differs only
negligibly because the controller remains near its lower bound.

### Phase 7 fresh confirmation

Phase 7 reruns five selected deployment conditions on independent replicas:

| Condition | Historical state | Role |
|---|---|---|
| Current only | None | No-history control |
| Fixed EWC | Rank-8 plus diagonal EMA, $\pi=.05$ | Compressed-history control |
| Hybrid B32 | FIFO replay of 32 plus fixed EWC archive | Principal constrained treatment |
| Replay B32 | FIFO replay of 32 | Direct bounded-replay comparison |
| Unbounded replay | All online arrivals | Unconstrained-memory control |

Every replica receives a fresh early-stopped $p=0$ fit and its own initial
Fisher estimate. The initial Fisher uses score samples at $p=0$, the six-sigma
Frobenius convergence rule, and a rank-8-plus-diagonal projection. It does not
use a future reference-optimum path. Conditions within a replica share that
initialization and ordered stream; replicas are statistically independent
apart from permitted overlap in the finite MNIST source data.

The generating process remains 100 evenly spaced $p$ values from zero to one
with eight arrivals per step. Principal plots and AUCs use only $p<.5$; this is
an analysis restriction, not a changed environmental schedule. Since metrics
at step $t$ are evaluated before its update, the model has observed exactly
$8t$ online arrivals at that point.

Confirmation proceeds in predeclared blocks of five fresh replicas, with ten
as the initial target and fifteen as the maximum. Pointwise trajectories and
paired differences use 95% Student-t intervals. The primary estimand is Hybrid
B32 minus Replay B32 environmental-accuracy AUC. Digit-9 OvR accuracy,
precision, and recall are reported together so adaptation and false-positive
behavior remain visible.

Ten fresh replicas met both predeclared precision targets. Hybrid B32 improved
environmental-accuracy AUC over Replay B32 by .0365 (95% CI [.0177, .0552])
while trailing its digit-9 OvR AUC by .0116 ([-.0209, -.0023]). The latter is
inside the $\pm.03$ practical-equivalence margin, but its sign is consistently
in replay's favor. EWC and Replay B32 both clearly outperformed current-only
learning. Hybrid B32 matched unbounded replay on environmental AUC, used 13.7
times less persistent state, and had better NLL and calibration; unbounded
replay retained higher digit-9 OvR accuracy and precision.

The initial-Fisher convergence check did not pass its stringent early-stop
criterion. Every replica used all 32,768 scores; the mean relative six-sigma
Frobenius radius was .230 versus a .01 target. Mean lag-one correlation was
near zero, and every Lanczos projection realized rank 8. These are therefore
maximum-budget high-sample estimates, not numerically exact Fisher controls.

## Plan 4 adaptive-composition study

Plan 4 asks whether an online Fisher-risk recommendation can adjust $\pi_t$
when the population path moves at a nonconstant rate. It retains the Plan 3
rank-8-plus-diagonal direct-EMA Fisher, $m=8$, no LFU, and the same EWC
objective semantics.

| Condition | Meaning | Evidentiary role |
|---|---|---|
| Fixed $.05$ | Constant composition available before Plan 4 | Prospective MNIST incumbent |
| Fixed $.025$ | Constant value identified during earlier analysis | Hindsight context only |
| Instantaneous Fisher risk | One-step plug-in local-surrogate recommendation | Noisy diagnostic; not promoted |
| EDR | Four-update EMA of the estimated Fisher-risk coefficients | Online recommendation under development |

The EDR experiments use a fixed `.05` cold start and $\pi_{\min}=.01$. On the
linear path, the recommendation eventually approaches the hindsight-useful
`.025` region. On logistic paths with $\kappa\in\{32,64,128,256\}$ it responds
to the speed event but becomes increasingly delayed and hysteretic. Applying
the recommendation closed loop underperforms fixed `.05` on all five
development paths.

The prequential ratio

$$
C_t=
\frac{\operatorname{EMA}_4(r_t^TG_tr_t)}
{\operatorname{EMA}_4\!\left[
\pi_t^2(q_{t-1}+m_t^{-1})\widehat D_{t\mid t-1}
\right]}
$$

is computable from one trajectory. It detects stale residual-risk forecasts
but does not establish predictive benefit. Plan 4 therefore rejects only the
tested EDR closed-loop treatment. Fixed $\pi$ is the incumbent for these MNIST
experiments, while adaptive composition remains open for applications in
which the locally appropriate weight changes and repeated tuning trials are
unavailable.
