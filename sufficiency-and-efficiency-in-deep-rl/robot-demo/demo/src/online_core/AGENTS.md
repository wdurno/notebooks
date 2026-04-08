# Oneline Core Specification 

This document provides instruction to the agent on how to implement the online core in `src/online_core/online_ssr_agent.py` and its supporting modules. 

## Goal 

Implement a sub-abstraction layer inheriting from `SSRAgent` in `src/core/ssr_agent.py`.
The `oneline_core` is an `SSRAgent` variant with an alternative `fit` function, 
leveraging sufficient statistics to update the agent in an online fashion. 
This streamlines user experience & VRAM utilization. 
The `fit` function uses the SSR regularizer to perform an `optimizer.step()` per every single observation taken, immediately after it is observed. 
Then, because the gradient vector is already cached, 
it can also be immediately memorized into the Fisher Information Matrix (FIM) sufficient statistic. 
Actions are sent to the robot asynchronously as fitting and memorization occur. 
This avoids having to regenerate or store VLM activations (which are large) 

## Motivation 

I've been running experiments using a typical RL paradigm: collect data, optimize, repeat. 
It's very slow, even with memorization, because recalculating gradients takes so much time. 
By embracing true online learning, all gradients only need to be calculated once per observation. 
All of this is enabled by storing a Fisher Information matrix $\mathcal I$ estimate, 
which need not be large thanks to LoRA and Krylov method-based estimates. 
So, for a reasonable increase in the cost to process each observation, 
we ultimately save in reprocessing all data per iteration, 
thus drastically reducing the overall computational cost from $O(n)$ to $O(p)$, 
where $n$ is the same size and $p$ is the tunable parameter count. 

## Mathematical argument 

The online core keeps its Fisher Information Matrix (FIM) estimate relevant with a score-only exponential moving average (EMA) process. 
This is the mathematically coherent object for our application: it is genuinely online, only requires the current observation's score, and admits a rigorous stochastic-process interpretation. 

Formally, the rigorous construction is indexed by a small parameter $\delta > 0$. 
As $\delta \to 0^+$, let $n_\delta \to \infty$ be the effective sample size induced by the EMA memory horizon, let $\pi_\delta \to 0$ be the EMA gain, and let $\pi_{k,\delta} \in [0,1]$ be the control variable used in the online update itself. 
In application we suppress $\delta$ and expose only a single user-facing argument `pi`, but the asymptotic argument below keeps these roles separate.

For score
```math
s_{k,\delta} := \nabla_\theta \ell(X_{k,\delta}; \theta_{k,\delta}),
```
define the EMA auxiliary score process
```math
\bar s_{k,\delta} = (1-\pi_\delta)\bar s_{k-1,\delta} + \pi_\delta s_{k,\delta},
\qquad
Y_{k,\delta} := \sqrt{\frac{2-\pi_\delta}{\pi_\delta}} \, \bar s_{k,\delta}.
```
Assume:
1. the score has uniformly bounded third absolute moment,
2. the drift field $b(\theta)$ and FIM $\mathcal I(\theta)$ are Lipschitz continuous, and
3. the parameter moves slowly across one EMA memory horizon, in the sense that
```math
\Delta_{k,\delta}
:=
\sum_{j \geq 0}
\pi_\delta (1-\pi_\delta)^j
\|\theta_{k-j,\delta} - \theta_{k,\delta}\|
=
O(\delta),
```
so the frozen-parameter approximation remains accurate locally.

Then, for frozen $\theta$, a weighted Lindeberg-Feller argument gives
```math
Y_{k,\delta} \Rightarrow \mathcal N(0, \mathcal I(\theta)),
```
and here is the key proof sketch. Under frozen $\theta$, the scores are iid with mean zero and covariance $\mathcal I(\theta)$. 
Unrolling the EMA gives
```math
Y_{k,\delta}
=
\sum_{j \geq 0} w_{j,\pi_\delta} s_{k-j,\delta},
\qquad
w_{j,\pi}
:=
\sqrt{\pi(2-\pi)}(1-\pi)^j.
```
These weights satisfy
```math
\sum_{j \geq 0} w_{j,\pi}^2
=
\pi(2-\pi)\sum_{j \geq 0}(1-\pi)^{2j}
=
1,
\qquad
\max_j |w_{j,\pi}|
=
\sqrt{\pi(2-\pi)}
\to 0
```
as $\pi \to 0$. So, for any fixed test vector $v$, the scalar projection
```math
v^T Y_{k,\delta} = \sum_{j \geq 0} w_{j,\pi_\delta} \, v^T s_{k-j,\delta}
```
is a weighted triangular array with total variance
```math
\sum_{j \geq 0} w_{j,\pi_\delta}^2 \, v^T \mathcal I(\theta) v
=
v^T \mathcal I(\theta) v.
```
If the score has bounded third absolute moment, then the Lindeberg condition follows from
```math
\sum_j
\mathbb E\!\left[
(w_{j,\pi_\delta} v^T s_{k-j,\delta})^2
\mathbf 1_{\{|w_{j,\pi_\delta} v^T s_{k-j,\delta}|>\varepsilon\}}
\right]
\leq
\frac{\max_j |w_{j,\pi_\delta}|}{\varepsilon}
\sup_{k,\delta}\mathbb E |v^T s_{k,\delta}|^3
\cdot
\sum_j w_{j,\pi_\delta}^2
\to 0.
```
Hence each projection converges to
```math
v^T Y_{k,\delta} \Rightarrow \mathcal N(0, v^T \mathcal I(\theta) v),
```
and the Cramer-Wold device yields the vector limit
```math
Y_{k,\delta} \Rightarrow \mathcal N(0, \mathcal I(\theta)).
```
Under the slow-motion assumption the same remains locally valid along the online trajectory up to approximation error $O(\sqrt{\pi_\delta} + \delta)$. 
So, although each update uses only one observation, the normalized EMA score behaves like a Gaussianized auxiliary process with covariance $\mathcal I(\theta_{k,\delta})$. 

This motivates the online update model
```math
\theta_{k+1,\delta}
=
\theta_{k,\delta}
+
\pi_{k,\delta} \frac{b(\theta_{k,\delta})}{n_\delta}
+
\sqrt{\frac{\pi_{k,\delta}}{n_\delta}} \, \mathcal I^{-1}(\theta_{k,\delta}) Y_{k,\delta},
```
where the covariance approximation inherits the same $O(\sqrt{\pi_\delta} + \delta)$ error. 
Equivalently, for a local displacement $d\theta_{k,\delta}$,
```math
\hat \theta_{k+1,\delta} - \theta_{k,\delta}
\approx
\mathcal N \left(
\pi_{k,\delta} d\theta_{k,\delta},
\;
\pi_{k,\delta} \mathcal I^{-1}(\theta_{k,\delta}) / n_\delta
\right).
```
This gives the same qualitative scaling as the large-batch EWC argument, but now with true single-observation memory usage. 

To construct the diffusion limit, define the triangular-array increments
```math
\Xi_{k,n_\delta}^{(\delta)}
:=
\pi_{k,\delta} \frac{b(\theta_{k,\delta})}{n_\delta}
+
\sqrt{\frac{\pi_{k,\delta}}{n_\delta}} \, \mathcal I^{-1}(\theta_{k,\delta}) Y_{k,\delta},
```
so that $\theta_{k+1,\delta} - \theta_{k,\delta} = \Xi_{k,n_\delta}^{(\delta)}$. 
With respect to the natural filtration $\mathcal F_{k,\delta}$,
```math
\mathbb E[\Xi_{k,n_\delta}^{(\delta)} \mid \mathcal F_{k,\delta}]
=
\pi_{k,\delta} \frac{b(\theta_{k,\delta})}{n_\delta}
+
o(n_\delta^{-1}),
```
and
```math
\mathrm{Cov}[\Xi_{k,n_\delta}^{(\delta)} \mid \mathcal F_{k,\delta}]
=
\pi_{k,\delta} \frac{\mathcal I^{-1}(\theta_{k,\delta})}{n_\delta}
+
O\!\left(\frac{\sqrt{\pi_\delta}+\delta}{n_\delta}\right).
```
Assume also that $\pi_{\lfloor n_\delta t \rfloor,\delta} \to \pi_t$ locally uniformly on compact time intervals. 
If $k = \lfloor n_\delta t \rfloor$, the maximal increment norm vanishes, the predictable drift sums converge to
```math
\sum_{j < n_\delta t}
\mathbb E[\Xi_{j,n_\delta}^{(\delta)} \mid \mathcal F_{j,\delta}]
\to
\int_0^t \pi_s b(\Theta_s)\,ds,
```
and the predictable quadratic variation converges to
```math
\sum_{j < n_\delta t}
\mathrm{Cov}[\Xi_{j,n_\delta}^{(\delta)} \mid \mathcal F_{j,\delta}]
\to
\int_0^t \pi_s \mathcal I^{-1}(\Theta_s)\,ds.
```
Thus standard triangular-array martingale convergence yields the diffusion limit
```math
d\Theta_t
=
\pi_t b(\Theta_t)dt
+
\sqrt{\pi_t}\,\mathcal I^{-1/2}(\Theta_t)dW_t.
```
Taking $\delta \to 0$ is what makes the frozen-parameter error disappear and turns the discrete EMA process into this continuous stochastic evolution.

The sufficient statistic should therefore be updated by EMA, not by a long-running arithmetic average. 
Define the per-observation Fisher contribution
```math
Z_{k,\delta} := s_{k,\delta} s_{k,\delta}^T,
```
and update
```math
\bar{\mathcal I}_{k,\delta}
:=
(1-\pi_\delta)\bar{\mathcal I}_{k-1,\delta} + \pi_\delta Z_{k,\delta}.
```
In implementation, $\bar{\mathcal I}_{k,\delta}$ will still be stored in the low-rank-plus-diagonal style already used by SSR, 
but it now estimates a recency-weighted FIM rather than a cumulative average over all past data. 
This exponential forgetting is essential: it keeps the statistic relevant while $\theta_{k,\delta}$ drifts, 
and it is what aligns the implementation with the stochastic-process model above. 
Operationally, this EMA behavior should be realized through an `mfi_alternate` operator passed into Lanczos rather than by rewriting the existing `memorize` routine. 

At step $k$, the fit loop processes a single observation and uses the current SSR regularizer built from $\bar{\mathcal I}_{k,\delta}$. 
The user may pass a fixed $\pi := \pi_{k,\delta}$, but the default should be the online optimal choice derived below. 
Let the true next generating point be
```math
\theta_{k+1,\delta}^* := \theta_{k,\delta} + d\theta_{k,\delta}.
```
Under the Gaussian approximation above,
```math
\mathbb E \| \hat \theta_{k+1,\delta} - \theta_{k+1,\delta}^* \|^2
\approx
\| (\pi_{k,\delta} - 1)d\theta_{k,\delta} \|^2
+
\pi_{k,\delta} \, \mathrm{tr}\left[\mathcal I^{-1}(\theta_{k,\delta}) / n_\delta\right].
```
Differentiating in $\pi_{k,\delta}$ gives the one-step MSE-optimal rule
```math
\pi_{k,\delta}^*
=
1
-
\mathrm{tr}\left[\mathcal I^{-1}(\theta_{k,\delta}) / n_\delta\right]
/
\left(2 \| d\theta_{k,\delta} \|^2 + \varepsilon \right),
\qquad
\varepsilon > 0.
```
In practice, clip $\pi_{k,\delta}^*$ to $[0,1]$ and estimate its ingredients from the online diagnostics already tracked by the agent. 
If the user supplies `pi`, use that value directly. 
Otherwise, use the clipped online estimate of $\pi_{k,\delta}^*$ as the default. 

This optimal rule changes the natural scaling of the controlled process. 
Write the true generating point as
```math
\theta_{k,n_\delta}^{*,\delta}
+
\frac{b(\theta_{k,n_\delta}^{*,\delta})}{\sqrt{n_\delta}},
```
so the observed recursion becomes
```math
\theta_{k+1,n_\delta}^{*,\delta}
=
\theta_{k,n_\delta}^{*,\delta}
+
\pi_{k,\delta}^* \frac{b(\theta_{k,n_\delta}^{*,\delta})}{\sqrt{n_\delta}}
+
\sqrt{\pi_{k,\delta}^* \, \mathcal I^{-1}(\theta_{k,n_\delta}^{*,\delta}) / n_\delta}\,\xi_{k,\delta},
\qquad
\xi_{k,\delta} \approx \mathcal N(0, I_p).
```
If $k = \lfloor \sqrt{n_\delta} t \rfloor$, then the accumulated drift remains order one while the accumulated covariance is only order $n_\delta^{-1/2}$. 
So the stochastic term vanishes asymptotically and the controlled path limit is
```math
d\Theta_t^*
=
\pi_t^* b(\Theta_t^*)dt.
```
For finite data we retain the noise and model the process by the small-noise SDE
```math
d\Theta_t^{\varepsilon,\delta}
=
\pi_t^* b(\Theta_t^{\varepsilon,\delta})dt
+
\sqrt{\varepsilon \pi_t^*}\,\mathcal I^{-1/2}(\Theta_t^{\varepsilon,\delta}) dW_t,
\qquad
\varepsilon = n_\delta^{-1/2}.
```
We do not implement either SDE directly. 
Their role in this document is to show that the EMA-based online design and the default $\pi_{k,\delta}^*$ rule are mathematically coherent, not merely heuristic. 

## Constraints

1. Do not specify a `loss` function in `OnlineSSRAgent` because it is still an abstract class. 
2. Do not modify anything in `src/core/`.
3. The data model should extend the original `SSRAgent` state, so we may load snapshots derived from `SSRAgent` instances.
4. Re-use existing `SSRAgent` attributes when possible to maintain single source of truth.

## Implementation tasks 

1. **`__get_get_grad_generator` variant**: `SSRAgent` recalculates gradients upon `memorize` calls, 
because storing them takes too much memory for large batches. 
Fortunately, online learning needs only apply a single, already cached gradient. 
For `OnlineSSRAgent`, please define `__get_get_grad_generator` to return `get_grad_generator` 
which returns `grad_generator`, a generator of one and only existing current gradient. 
It's assumed that gradients have already been calculated. 
If they haven't, throw an error. 
2. **`act` function**: `OnlineSSRAgent` needs an abstract `act` function used to send instructions to the agent mid fit loop. 
While `act` is allowed to have useful side effects (like populating the replay buffer), 
the abstract definition returns a reward to be used in fitting. 
This allows all activations to be produced precisely once per action and optimization.
3. **`mfi_alternate` EMA rule**: Do not rewrite `SSRAgent.memorize`. 
Instead, provide an `mfi_alternate` function for use with `src/core/lanczos.py:l_lanczos`. 
This function should encode the EMA-weighted Fisher update rule while preserving the existing `memorize` machinery. 
Concretely, if the current observation contributes score $s_t$ and therefore $Z_t = s_t s_t^T$, then the operator used by Lanczos should act like the recency-weighted FIM
```math
\bar{\mathcal I}_t
=
(1-\pi)\bar{\mathcal I}_{t-1} + \pi Z_t.
```
So, for a vector $x$, `mfi_alternate(x)` should represent multiplication by the EMA-updated Fisher estimate rather than by the cumulative-average Fisher estimate. 
This keeps the implementation aligned with the online theory while reusing the existing low-rank Lanczos pipeline. 
4. **`fit` function**: Override the existing `SSRAgent.fit` function in `OnlineSSRAgent`. 
Have these arguments:
   1. `iters` defines how many times the current observation is iterated upon. Zero is acceptable, since sometimes we're only memorizing.
   2. `pi` is an optional float override. If omitted, compute the clipped online estimate of $\pi^*$ from the diagnostics implied above.
   3. `memorize` (boolean) determines whether the EMA-aware Fisher memorization is applied. If `True` and `iters > 1`, only memorize on the final iteration.
   4. `grad_clip` remains optional as in the base class.
The fit loop should:
   1. obtain one new observation via `act`,
   2. compute the single-observation loss,
   3. combine the immediate loss with the SSR regularizer using the current `pi`,
   4. backpropagate and apply `optimizer.step()`, and
   5. if `memorize` is enabled, call the existing memorization pathway with the EMA-aware `mfi_alternate` behavior so the Fisher estimate stays recency-weighted without recomputing activations.
