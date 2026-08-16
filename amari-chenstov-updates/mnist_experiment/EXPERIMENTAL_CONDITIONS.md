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
