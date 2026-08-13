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
no-EWC deployment benchmark.
