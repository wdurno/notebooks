# Implementation Plan 18: Combined Memory, Fisher Transport, And Adaptation

**Motivation:** improve continual learning with limited new observations,
replay capacity, and compute. Plan 15 found useful sample savings from an
age-dependent STM sensitivity penalty; Plan 16 found an applied benefit from
paired diagonal Fisher corrections; Plan 17 eventually found useful
velocity-Kalman adaptation under sustained, continuous drift. The next question
is whether these contributions remain useful together, after controlling
retention, rather than whether their historical gains simply multiply.

**Status:** Phases 0-4 are complete. Execution was authorized on 2026-11-06;
the completed study contains 33 development paths and 40 fresh/stationary paths
(six changing and two stationary paired replicas, all five arms). Recovery,
CPU/CUDA acceptance and the final full-checksum/pairing audit passed. Total
charged compute, including final tests and reporting, is about 7.38 hours.
See the execution record below and `PLAN18_HANDOFF.md`. All five selected arms
met the development matching tolerance; fresh retention support is assessed
separately in the notebook. The mathematical
extension below belongs here until its assumptions and applied consequences
have been assessed; leave `mathematical_overview.ipynb` unchanged.
**Phase 5:** complete. The broader continuous-drift extension finished all 77
trajectories: 37 development, 30 fresh, and 10 stationary. Six fresh and two
stationary paired blocks passed the final integrity audit. Post-reboot checks
on 2026-11-08 verified all 107 completed units with full checksums and confirmed
that no resumption is needed. Total cumulative charge across both protocols,
including the follow-up audit and reporting, is about 14.58 active hours.
`PLAN18_HANDOFF.md` records completion and qualifications. Unavailable notebook
estimates are labelled "not estimable"; they are not unfinished computations.
**Phase 5 scientific qualification:** retention calibration failed. The five-arm
confirmation is not valid as a comparison at comparable retention; contrasts
involving B and the factorial interaction are confounded by its much stronger
ridge. See the Phase 5 conclusion below. Recorded results remain immutable.
**Phase 6:** complete on 2026-11-08. Four-member development calibration,
all 30 confirmation trajectories, and the final full-checksum/pairing audit
passed. All ten fresh pairwise contrasts meet both nominal .10-pixel retention
equivalence checks. The combined policy improves new-domain risk; STM alone
does not improve it in this regime. Cumulative charge is about 19.58 hours.
The user subsequently authorized a conditional 21.5-hour cumulative ceiling,
only if calibration passes and the original allowance cannot fit confirmation.
**Phase 7:** drafted for review, not executing. Fix the STM incumbent B at
ridge ratio 1, recalibrate A/C/D/E to its retention, and run a larger fresh
comparison. Preserve Phase 6 as a valid result at a different operating point.
The proposed additional compute allowance is stated in Phase 7; it does not
retroactively change the historical Phase 6 authorization.
**Agreed controller boundary:** retain Plan 17's velocity-Kalman observer and
weight rule. Do not estimate the learner-error cross moment $C$ online or add
a correlation-corrected controller to this experiment.

**Report:** `mnist_experiment/plan18_results.ipynb`, concise and
human-readable, with progress near the top and results in execution order.
**Operating model:** Astra develops the theory and design; after approval,
Sol implements, validates, calibrates, executes, and records material repairs.
**Authorized resource ceiling (Phases 0--6):** originally 20 cumulative active wall-clock compute hours,
extended by explicit user approval on 2026-11-08 to at most 21.5 hours after
passing Phase 6 calibration, without changing the confirmation allocation,
and 500 GB of artifacts, including failed attempts and reporting. This is a
compute allowance, not a measured runtime forecast.

## Executive Summary

- **STM penalty:** make older predictions less dependent on coordinates that
  will not be protected by the LTM archive.
- **Paired FD transport:** reduce parameter-location staleness in the archived
  LTM Fisher diagonal, without adding replay observations to that archive early.
- **Adaptive weight:** estimate when new evidence deserves more weight relative
  to retained information. Preserve velocity Kalman, observing fresh arrivals
  once each, with no online cross-moment estimator or discrepancy correction.
- **Comparison:** use the same model and replay resources in every arm, then
  calibrate retention using development data only. Measure new-data performance,
  retention, samples saved, and actual compute together.

The handoff supplies a particularly useful explanation: older observations
should become representable by the kind of compressed memory we actually keep.
That is a stronger mathematical starting point than equating lower STM Fisher
trace with demonstrated information transfer or reduced model dimension.

## Scope And Provenance

The starting task is **nine-digit grid registration from Plan 15**, with the
sustained rotation regimes motivated by Plan 17. Plan 16 supplies the paired
FD method, not the historical grid task. Retain an applied regression target:
three two-dimensional landmarks per digit, hence 54 scalar outputs per grid.

The supplied *STM/LTM theory and statistical-efficiency handoff* inspected
commit `445d19762458e889178dedad3fb65d53ce278797`. This draft was reconciled with
`b0bdcca`, particularly the grid learner, STM-capacity model, and Plans 16--17.
Important inherited facts and deliberate changes are:

| Item | Starting point or proposed change |
|---|---|
| STM placement | Plan 15's head-only rank-one adapter; both factors trainable |
| Capacity | Existing model: 3,268 LTM and 108 STM coordinates, 3,376 total |
| LTM learning | All base parameters remain trainable; this is not frozen-base LoRA |
| Likelihood | Gaussian landmark likelihood with fixed covariance $4I_{54}$ |
| Replay | FIFO32, replay2, equal incoming/replay mean-loss weights, and nominal arrival unit $m_0=4$ grids |
| EWC representation | Strictly diagonal in every arm; historical Plan 15 used rank-64-plus-diagonal |
| Archive admission | Initialization summary, then each online observation exactly once, on eviction |
| Fisher correction | Paired endpoint differences along actual accepted displacement, LTM output block only |
| Action | Fixed $\pi=.025$ controls versus predictable velocity-Kalman adaptation |

Historical effects are motivation, not expected effect sizes for this changed
combination. In particular, the diagonal-only incumbent must be evaluated
afresh. Use 40-pixel padded cells and cell-aligned pooling as specified below,
retaining the feature-map/head dimensions and asserting actual parameter counts.

## Theory: One Likelihood, Two Memory Blocks

### Predictive And Optimization Coordinates

Write $\theta=(\theta_L,\theta_S)$, with negative log likelihood $L$,
$g=\nabla L=-s$, and the single likelihood Fisher

$$
\mathcal I(\theta)=\mathbb E_\theta[s s^T].
$$

For conditional prediction its per-input integrand is

$$
F_x(\theta)=\mathbb E_{Y\sim p_\theta(\cdot\mid x)}
[s(x,Y;\theta)s(x,Y;\theta)^T].
$$

The LTM and STM blocks are blocks of this geometry, not separate Fisher
estimands. Input-cohort averages are estimators or explicit retention
diagnostics. They do not introduce an independent distribution argument into
$\mathcal I$. Observed-label scores are used for training and observing drift;
model-label integration is used for Fisher measurements.

With $W_{\rm eff}=W_L+\gamma BA$, let $w=m(\theta)$ denote the merged base
parameters. Since the base weights also train,

$$
p_\theta=\bar p_{m(\theta)},\qquad
\mathcal M_{L+S}=\mathcal M_L,\qquad
\mathcal I(\theta)=Dm(\theta)^T\bar{\mathcal I}(w)Dm(\theta).
$$

The merge derivative has an identity block in the base coordinates. Thus
$\operatorname{rank}\mathcal I=\operatorname{rank}\bar{\mathcal I}\le p_L$,
although the raw chart has $p_L+p_S$ coordinates. STM adds optimization freedom,
not a larger unrestricted predictive family. Unlike the original
[LoRA construction](https://arxiv.org/abs/2106.09685), our base is not frozen.

One prediction-preserving tangent is
$\delta W_L=-\gamma(\delta B A+B\delta A)$. Consequently the joint Fisher is
singular even before ordinary neural-network symmetries are considered.
Do not apply an inverse-Fisher/parameter-count efficiency formula to the raw
joint chart. Use merged coordinates for the observer and local predictive risk;
they remove this particular redundancy, not necessarily every nonidentifiability.
The maintained EWC remains diagonal in the fixed LTM implementation chart.

### The Objective We Intend To Preserve

Let $B_t$ contain new observations, $R_t$ the sampled replay observations,
$\alpha_t$ the LTM archive anchor, and $D_t=\operatorname{diag}(d_t)$ its
maintained diagonal. Freeze all of these during a solve. Set

$$
\omega_t=\frac{1-\pi_t}{\pi_t},\qquad
s_t=\frac{\operatorname{tr}D_t}{p_L},\qquad
\kappa_t=c_{\rm ridge}s_t.
$$

The shared objective is

$$
\begin{aligned}
J_t(\theta)={}&(1-\beta)\bar L_{B_t}(\theta)
+\beta\bar L_{R_t}(\theta)
+\frac{\omega_t}{2}(\theta_L-\alpha_t)^TD_t(\theta_L-\alpha_t)\\
&+\frac{\omega_t\kappa_t}{2}
\left(\|\theta_L-\alpha_t\|^2+
\|\theta_S-\theta_{S,t}\|^2\right)
+\lambda_F\mathcal C_t(\theta).
\end{aligned}
$$

With an empty buffer, use the incoming mean loss alone and no STM penalty.
The STM ridge anchor is the pre-arrival STM state. Ridge is numerical/statistical
regularization, not archived likelihood evidence. Record $s_t$, $\kappa_t$,
and the resulting penalty strengths; LFU can change ridge strength through
$s_t$ even with a fixed ridge ratio. This inherited coupling is part of the
primary algorithm, not an isolated test of curvature orientation.

For squared clipped ages $a_{i,t}$, the ideal sensitivity term is

$$
\mathcal C_t(\theta)=\frac{1}{|R_t|p_S}
\sum_{i\in R_t}a_{i,t}\operatorname{tr}F_{SS,x_i}(\theta).
$$

Use the existing symmetric FD-KL approximation, with fresh named-seed probes
per solve but fixed probes within its optimizer closures:

$$
\Phi_h(x,\theta,v)=
\frac{D_{\rm KL}(p_{\theta+hE_Sv/2}\|p_{\theta-hE_Sv/2})
+D_{\rm KL}(p_{\theta-hE_Sv/2}\|p_{\theta+hE_Sv/2})}{h^2},
\quad \mathbb E[vv^T]=I.
$$

Under local smoothness, $\mathbb E_v\Phi_h=
\operatorname{tr}F_{SS,x}+O(h^2)$. Average probes, then observations with age
weights, and divide by $p_S$, not output dimension. Both model evaluations
remain differentiable. All replay loss gradients may update both memory
blocks; the FD term updates every block on which it mathematically depends.

For the inherited additive, head-only Gaussian adapter, its FD sensitivity has
no direct derivative with respect to the additive base head weights. It can
still influence LTM encoder parameters and the subsequent replay optimization.
This is not a requirement to detach LTM, nor a proof of a particular transfer
mechanism. Preserve the successful chart rather than silently replacing it.

### Why Low STM Sensitivity Can Help An LTM Archive

Fix an eviction-time predictive reference $\theta_c$ and a bounded subsequent
move $u=(u_L,u_S)$. For a fixed input or fixed weighted cohort, local predictive
KL has the expansion

$$
K_c(u)=\tfrac12u^TF_cu+r_c(u),\qquad
|r_c(u)|\le c_3\|u\|^3.
$$

An LTM-only archive approximates it by
$\widetilde K_c(u)=\tfrac12u_L^TDu_L$. Even a perfect LTM block omits
$u_L^TF_{LS}u_S+\tfrac12u_S^TF_{SS}u_S$. Positive semidefiniteness gives

$$
|u_L^TF_{LS}u_S|
\le\sqrt{u_L^TF_{LL}u_L}\sqrt{u_S^TF_{SS}u_S}.
$$

If $\|F_{LL}\|_{\rm op}\le M_L$, $\operatorname{tr}F_{SS}\le\epsilon_S$,
$\|D-F_{LL}\|_{\rm op}\le e_L$, and $\|u_L\|\le r_L$, $\|u_S\|\le r_S$,
then

$$
\boxed{
|K_c(u)-\widetilde K_c(u)|\le
\tfrac12e_Lr_L^2+r_Lr_S\sqrt{M_L\epsilon_S}
+\tfrac12\epsilon_Sr_S^2+c_3\|u\|^3.
}
$$

This is the main proposed integration of the handoff into our theory:

| Contribution | Error it can address | What it does not establish |
|---|---|---|
| Age-weighted FD KL | Omitted STM and cross-block terms through smaller $\epsilon_S$ | Accurate eviction predictions or actual transfer by itself |
| Paired FD LFU | Parameter-staleness contribution to $e_L$ | Recovery of missing off-diagonal curvature or discarded input populations |
| Adaptive $\pi$ | Balance of retained-estimate error against fresh-evidence noise | Global optimality of the nonlinear learner |

The trace penalty controls an average on sampled replay inputs. It does not
guarantee a uniform bound for every evictee or future displacement. Likewise,
LFU estimates a correction and can increase rather than decrease realized error.
The displayed bound is local at a common reference, not an exact expression
for a long-running, recentered archive.

For diagonal archives, $e_L$ includes irreducible representation error:

$$
\|D-F_{LL}\|_{\rm op}
\le\|D-\operatorname{Diag}(\operatorname{diag}F_{LL})\|_{\rm op}
+\|\operatorname{Diag}(\operatorname{diag}F_{LL})-F_{LL}\|_{\rm op}.
$$

LFU targets part of the first term, not the second. These dense matrices are
theoretical objects; the experiment does not need to materialize them.

The practical hypothesis is therefore **more faithful, more current compressed
protection**, not guaranteed additive gains. A small STM trace is also
chart-dependent: compensating LTM/STM changes can preserve predictions while
reducing that trace. Preserve the chart, check predictive quality, and do not
interpret trace as a count of remembered observations.

### A Separate Route To Sample Efficiency

The handoff's scalar calculation shows why improved retention calibration can
save data without increasing information in each observation. Let a new
estimate have variance $\sigma^2/n$ and an independent old anchor have MSE $v$
relative to the current target. Combining them with new-evidence weight $k$
gives

$$
R(k)=(1-k)^2v+k^2\sigma^2/n,\qquad
k^*=\frac{v}{v+\sigma^2/n}.
$$

For summed Gaussian likelihood plus retention precision $\Lambda$, the optimal
precision is $1/v$, not automatically the inverse historical variance. For
$\sigma^2=1$, $v=.01$, precision 400 needs 200 new observations to reach MSE
.005, whereas precision 100 reaches it at 100 observations. This is an exact
constructed example, not a prediction of 50% savings in Plan 18.

STM can also change effective regularization without expanding the predictive
family. Locally write an effective displacement as $u=l+Ts$. For positive
definite penalties $P$ and $R$, profiling
$\tfrac12l^TPl+\tfrac12s^TRs$ over this constraint gives

$$
P_{\rm eff}=P-PT(R+T^TPT)^{-1}T^TP
=(P^{-1}+TR^{-1}T^T)^{-1}\preceq P.
$$

This is a local quadratic illustration, not the full nonlinear FD objective.
It applies to the replay control with the same STM too. It cannot alone explain
the incremental FD-KL benefit. Development retention matching is essential
precisely because several interventions can alter effective protection.

## LFU Transport Without Premature Consolidation

Maintain two different references: $\alpha_t$ is the **retention center**;
$\varphi_t$ records the **parameter location of the transported curvature**.
Moving curvature to a new location does not admit an observation to EWC or
justify moving its retention center.

Let $d_L(\theta)$ be the LTM diagonal of the single Fisher. Its derivative is
the corresponding diagonal projection of the full LFU:

$$
Dd_L(\theta)[u]=\operatorname{diag}_L[(C_\theta+R_\theta):u].
$$

Here $u$ includes **both** LTM and STM movement. The output is LTM-only, not
the direction of differentiation. For a Gaussian regressor with fixed
covariance $\Sigma$,

$$
[F_{LL,x}]_{ii}=(\partial_{L,i}f_\theta(x))^T
\Sigma^{-1}(\partial_{L,i}f_\theta(x)).
$$

Integrate model labels analytically; squared observed residuals are not a
substitute. Even here the head-block Fisher may not change when only head
weights move and encoder features stay fixed. Such a zero correction is a
valid geometric result, not a numerical failure.

### Transaction Order

Use a post-solve transport transaction to avoid correcting a newly admitted
Fisher observation twice. This ordering is an explicit extension of Plan 16's
no-replay learner, not an unannounced change to its frozen implementation.

1. Before batch $B_t$, choose $\pi_t$ from past information. Freeze it for the
   objective and this transaction's retirement weighting.
2. Stage FIFO updates while all prospective evictees remain available. Solve
   the objective using the pre-existing archive and accepted state $\theta_t$.
3. After accepting $\theta_{t+1}$, measure the old archive's parameter correction
   on identical inputs at $\theta_t$ and $\theta_{t+1}$.
4. Blend in actual evictees' Fisher only after this correction. Recenter the
   retention anchor only if there are evictees. Commit the entire transaction
   atomically, including FIFO state, archive, controller, and unique-ID ledger.

Measurement inputs $Q_t$ are the evictees when nonempty; otherwise
use the incoming batch. They are still available training inputs, never a
holdout or a resurrected discarded buffer. With
$z_{Q,L}(\theta)=|Q|^{-1}\sum_{x\in Q}\operatorname{diag}F_{LL,x}(\theta)$,

$$
\Delta_t=z_{Q_t,L}(\theta_{t+1})-z_{Q_t,L}(\theta_t).
$$

The endpoint identity is

$$
d_L(\theta_{t+1})-d_L(\theta_t)
=\int_0^1 Dd_L(\theta_t+a u_t)[u_t]\,da.
$$

Call this **paired FD transport**, a finite-change estimator of the integrated
full LFU, rather than an exact pointwise linearization. Matched inputs reduce
difference noise. Their suitability for transporting the historical archive
remains a local representativeness assumption; no endpoint identity removes
input-sampling bias. Do not claim the correction exactly recovers old-cohort
curvature after its inputs have disappeared.

For $e_t=|E_t|$ evictees and nominal arrival unit $m_0$, set

$$
\eta_t=1-(1-\pi_t)^{e_t/m_0},\qquad
d_{t+1}=\Pi_+\left[(1-\eta_t)(d_t+\chi\Delta_t)
+\eta_t z_{E_t,L}(\theta_{t+1})\right],
$$

where $\chi=1$ for transport arms and 0 otherwise; omit the final term when
$e_t=0$. $\Pi_+$ clips negative diagonal entries to zero. Log negative mass,
nonfinite values, correction magnitude, and projection size before clipping.
Accumulate endpoint differences in float64 and test cancellation at small moves.

If there is no eviction, $\eta_t=0$, $\alpha_{t+1}=\alpha_t$, and no archive
evidence weight is added. A transport arm may still change $d_t$ and advance
$\varphi_t$ to $\theta_{t+1}$. On eviction, set
$\alpha_{t+1}=\theta_{L,t+1}$ as in Plan 15. The FD-KL STM term is evaluated
anew at every candidate $\theta$; it has no archived matrix to transport.

This EMA and accepted-state recentering remain an **approximate retention
statistic**, not exact Gaussian natural-parameter accumulation. Updating
curvature alone does not restore the historical loss gradient or repair a
changed center. The distinction is consistent with the need for careful
quadratic bookkeeping highlighted by
[Huszar's EWC analysis](https://arxiv.org/abs/1712.03847), although that paper
does not prove this proposed transport algorithm.

## Recovering An Optimal Adaptation Concept

### A General Local Blend Includes Dependence

**Theoretical scope only:** this subsection describes how dependence can alter
an ideal blend. It does not specify another implemented controller or require
estimating $C$ in development or confirmation. The experiment retains the
velocity rule below and evaluates whether that approximation transfers usefully.

Work in effective predictive coordinates and choose a fixed positive
semidefinite local risk metric $M$. Under one specified local probability law,
let $a$ be an old estimate and $z$ a new-evidence proposal for the same target
$w^*$. Define

$$
e_o=a-w^*,\quad e_n=z-w^*,\quad
A=\mathbb E\|e_o\|_M^2,\quad B=\mathbb E\|e_n\|_M^2,\quad
C=\mathbb E[e_o^TMe_n].
$$

These are error second moments, including bias. $C$ is not just a centered
covariance, nor the position-velocity covariance already propagated inside the
Kalman filter. Correlated-estimate fusion is compatible with Kalman theory;
knowing the requisite error moments is the difficulty here. For an affine blend
$\widehat w(k)=(1-k)a+kz$,

$$
\mathcal R(k)=(1-k)^2A+k^2B+2k(1-k)C,
$$

$$
\boxed{k^*=\operatorname{clip}_{[0,1]}
\frac{A-C}{A+B-2C}.}
$$

The denominator equals $\mathbb E\|e_n-e_o\|_M^2\ge0$. If it is zero, the
two proposals are indistinguishable in this risk metric and every blend has
the same risk. Otherwise the formula is the exact minimizer of this specified
quadratic risk. No joint inverse Fisher or unbiased-MLE assumption is needed.

Recover the overview's applied rule by assuming a centered old tracking error,
uncorrelated old/new errors, matched covariance shapes, and matched quadratic
curvatures, so $k=\pi$ and

$$
A=\mathcal S_t+q_tV_t,\qquad B=V_t/m_t,\qquad C=0,
$$

$$
\pi_t^*=\frac{\mathcal S_t+q_tV_t}
{\mathcal S_t+q_tV_t+V_t/m_t}.
$$

Here $\mathcal S_t$ denotes movement energy, distinct from the sample-efficiency
score $S$ reported later. Under biased historical tracking, replace movement
energy by the appropriate squared displacement from the mean old error.

Use one conditioning convention throughout a calculation. If conditioning
already fixes the old error, do not add its replica covariance $q_tV_t$ again.
Likewise, a filtered posterior uncertainty and a historical covariance proxy
are not automatically independent sources of uncertainty. The existing
velocity rule remains a calibrated marginal-risk approximation.

### Why Replay Needs Separate Bookkeeping

Repeating an observation does not make an independent new score. In a local
linear influence approximation, write estimator error as

$$
e\simeq b+\sum_{j\in\text{unique observations}}H_j\xi_j.
$$

For fixed coefficients and independent centered $\xi_j$, covariance is
$\sum_jH_j\Omega_jH_j^T$. Multiple uses of observation $j$ must be added
inside $H_j$ before this product. Adaptive coefficients and dependent inputs
require further calibration or cross terms. A replay batch of size $b$ does
not turn $m$ fresh observations into $m+b$ independent observations.

The archive does have a separate exact nominal concentration recurrence for
disjoint, uniformly weighted eviction groups:

$$
q^E_{t+1}=(1-\eta_t)^2q^E_t+\eta_t^2/e_t\quad(e_t>0).
$$

With no eviction it is unchanged. Transport adds no independent evidence and
does not change this concentration. Initialization contributes its own recorded
weights. This identity is bookkeeping, not a claim that LFU corrections or
learner covariance follow the same scalar weights.

To preserve the first controller comparison with Plan 17, also carry its
**reference composition state**

$$
q^{\rm ref}_{t+1}=(1-\pi_t)^2q^{\rm ref}_t+\pi_t^2/m_t.
$$

Initialize both nominal states at $1/n_0$ for a uniformly weighted summary of
$n_0$ independent initialization grids, or record the actual initial weight
concentration if that assumption differs. Count grids, not their 54 outputs.

Use this state in the initial velocity rule, explicitly labeled a controller
proxy. It is exact for its declared virtual arrival weights, **not** the
actual replay learner's effective sample size and **not** the archive's
eviction-only weight state. Measuring how well it transfers is part of
development. Do not silently replace $m_t$ by replay passes or scalar labels.

### EWC Odds Are Not Automatically An Effective Gain

Even without replay dependence, positive definite local data and retention
curvatures $H$ and $P$ on the working predictive subspace produce

$$
\widehat w(\pi)=(H+\omega P)^{-1}(Hz+\omega Pa),\qquad
\omega=(1-\pi)/\pi.
$$

The effective gain is the matrix $(H+\omega P)^{-1}H$, not generally $\pi I$.
If $P=H$, it is $\pi I$. If $P=a_HH$ for a positive scalar $a_H$, then

$$
k=\frac{\pi}{\pi+a_H(1-\pi)},\qquad
\pi=\frac{a_H k}{1-k+a_H k}.
$$

Thus a scalar optimal gain still maps cleanly to an EWC action in this special
case. STM profiling, different anchors, the FD penalty, and unmatched
curvatures generally destroy the simple equality. Even a diagonal raw EWC
does not make the profiled effective curvature diagonal.

For the actual learner, the defensible local optimum is instead

$$
\pi_t^{*,\rm local}=\arg\min_{\pi\in[\pi_{\min},\pi_{\max}]}
\mathbb E\!\left[
\|m(\widehat\theta_{t+1}(\pi))-w_{t+1}^*\|_M^2
\,\middle|\,\mathcal G_t\right],
$$

with the conditioning information $\mathcal G_t$ and local estimator model
specified. The oracle target is not available online. The velocity-Kalman
formula is an economical approximation to this concept, not its proven
solution in the combined architecture.

**Agreed implementation:** retain the working velocity formula and reference
composition state for this study. Do not implement an estimator of $C$, a
correlation-corrected gain, or a replay-aware influence model. Such work would
require a separately approved extension, not an autonomous calibration repair.
The generalized formula explains possible approximation error; the bounded
development response audit assesses the controller's practical transfer without
claiming to recover the unknown error moments.

### Observer In The Combined Model

At a common effective-parameter anchor $a$, use incoming observed-label
scores only:

$$
y_t=D_{a,\epsilon}^{-1}\frac1{m_t}
\sum_{(x,y)\in B_t}\nabla_a\log p_a(y\mid x).
$$

$D_{a,\epsilon}$ is a numerically floored diagonal approximation to the
anchor Fisher; it is not an inverse of the singular joint STM/LTM Fisher.
Keep the anchor fixed over the declared observer interval. The approximation
$\mathbb E\bar s(a)\simeq\bar{\mathcal I}(a)(w^*-a)$ remains local and need
not be diagonal. Record its calibration and weak-coordinate diagnostics.

Each incoming batch supplies exactly one observer measurement. Replay passes,
optimizer closures, and LFU endpoint evaluations supply none. Under the stated
observation model, with latent state $x_t$ and past information $\mathcal F_{t-1}$,
fresh measurement noise satisfies

$$
\mathbb E[\nu_t\mid\mathcal F_{t-1},x_t]=0.
$$

The prior observer error $\widehat x_{t\mid t-1}-x_t$ is measurable given those
quantities, so its cross moment with $\nu_t$ vanishes by iterated expectation.
This preserves the ordinary Kalman construction under its assumed dynamics
and noise model without fitting an additional cross moment. Exact filtering
optimality still requires the correct linear-Gaussian model and covariances;
the score map, diagonal approximation, and estimated noise can violate that
idealization. Fresh-only measurements do **not** make the complete replay/STM
learner's $C$ zero or prove its EWC action optimal.

Retain Plan 17's per-coordinate position/velocity state on the incoming-batch
clock, its innovation-noise estimation, and its movement second moment:

$$
\mathcal S_t=\sum_i d_{t,i}
(\widehat v_{t,i}^{2}+P_{vv,t,i}+Q_{zz,t,i}),\qquad
V_t=\sum_i d_{t,i}K_{t,i}.
$$

The implemented recommendation remains

$$
\pi_t^{\rm rec}=\frac{\mathcal S_t+q_t^{\rm ref}V_t}
{\mathcal S_t+q_t^{\rm ref}V_t+V_t/m_t},
$$

subject to the declared cold start and safeguards. It contains no estimated
$C$. Ordinary Kalman state covariances, including position-velocity covariance,
and the inherited measurement-noise estimation remain in scope; excluding a
learner cross-moment estimator does not remove them.

$K_{t,i}$ is the estimated one-observation noise scale after the anchor map;
the filter measurement variance divides it by $m_t$. Filter gains and the
learner action $\pi_t$ are different quantities. All terms used for $\pi_t$
must be available before $B_t$. Scores from $B_t$ affect only later actions;
replay never advances the observer's observation clock.

Start development with the inherited 16-arrival cold start at .025 and
$[10^{-4},.2]$ safeguards, then freeze the validated settings before fresh
replication. Do not feed true rotation, regime boundaries, holdout outcomes,
or future labels to the observer. Calibration should include fixed-anchor
linearization error; replacing the observer during fresh replication requires
a separately versioned cohort.

## Experiment

### Five Conditions

All five use the same head-rank-one STM/LTM model, diagonal EWC, FIFO capacity,
replay sampling, and optimizer budget. A replay control includes trainable
STM; it is not a reduced-capacity model.

| ID | Report label | Age FD KL | Paired FD transport | Adaptation weight |
|---|---|---|---|---|
| A | Replay/EWC | No | No | Fixed .025 |
| B | STM incumbent | Yes | No | Fixed .025 |
| C | STM + transport | Yes | Yes | Fixed .025 |
| D | STM + adaptation | Yes | No | Velocity Kalman |
| E | Combined | Yes | Yes | Velocity Kalman |

Primary questions are E versus B (value beyond the STM incumbent) and E versus
A (total improvement over replay/EWC). C and D identify incremental transport
and adaptation effects on that background. For a lower-is-better outcome
$\ell$, the conditional interaction is
$\ell_E-\ell_C-\ell_D+\ell_B$; a negative value indicates more improvement
than the sum of these two increments on this scale.

This is not the full eight-arm factorial and cannot estimate every interaction
with FD KL. Do not add that factorial automatically. The frozen initial model
is an evaluation reference, not a sixth trained arm. Do not weaken the fixed
control to a conveniently smaller $\pi$.

### Continuous Regimes And Holdouts

An incoming batch contains four previously unused grids. Use 160 initial
quiet batches, 304 outbound batches, 304 return batches, and 160 final quiet
batches: **928 batches / 3,712 grids / 33,408 digit pairs**. A grid contains nine
different source digits, with independently drawn cell transforms. The incoming
batch is the learner and observer clock; do not split it at novel observations.

Set $h_t=0$ in the quiet intervals. On outbound batch $j=1,\ldots,304$, set
$h_t=(1-\cos(\pi j/304))/2$; on return batch $j=1,\ldots,304$, use
$h_t=(1+\cos(\pi j/304))/2$. There is no central plateau. The continuous
interpolation has zero slope at its endpoints and turning point.

Independently label each incoming grid novel with probability
$p_t=.125h_t$. Nominal cells have rotation distributed as a normal with location
zero and standard deviation 6 degrees, truncated to the fixed support
$[-70,70]$ degrees. Novel cell $j$ has the same law with location
$s_j45h_t$ degrees. For each replica, randomly permute a fixed four-positive,
five-negative sign pattern across cells, then apply a balanced replica-wide
sign flip. Keep that pattern throughout the trajectory and its holdouts.
Do not redraw signs per observation and inadvertently cancel the coherent
direction being tracked.

The peak **location** is 45 degrees; it is not a hard 45-degree support cutoff.
The fixed truncation support avoids a changing hard boundary in the angle law.
Show the densities and the realized/expected schedule, including how often
novel-component draws overlap the nominal angle range. Nuisance translations
are independently uniform on $[-2,2]$ pixels, log-scales on
$[-\log 1.1,\log 1.1]$, and shear on $[-.1,.1]$, in the inherited forward-affine
order. Keep these laws identical in all regimes. Do not also introduce Plan
15's changing blur severity: this study changes rotation and its mixture weight.

The schedule yields **152 expected novel grids**, or 1,368 novel digit pairs,
over the full trajectory. The common sample-efficiency analysis budget is the
first **100 novel grids**, not an outcome-dependent stopping time for training.
Finish the return and final quiet interval even after that count is reached.
Publish actual counts and later acquisition/retention separately. Under these
Bernoulli draws, the probability of fewer than 100 novel grids is approximately
$1.10\times10^{-6}$; nevertheless retain such a seed, report its unsupported
inverse statistics, and never redraw it.

The maximum rotation-location increment is about .233 degrees per batch,
matching Plan 17's successful ramp speed. Each quiet interval spans about
5.84 fixed-.025 weight half-lives and five measurement-noise half-lives.
These are design checks, not a guarantee that every adaptive archive settles.
Measure motion/noise, quiet-interval adaptation, and filter lag. Smoothness and
these durations do not by themselves prove LAN. Do not shorten regimes to
fit more replicas or expose schedule metadata to an online policy.

Fresh stationary replicas use the same length and source-budget structure but
$h_t=p_t=0$ throughout. Their novel holdout measures untrained-domain
generalization, not acquisition; no novel-sample $S$ is defined for them.

### Sources, Rendering, And Initialization

Allocate the following distinct official-training source images **within each
replica**, before any tuning. Conditions share the exact allocation. Separate
replicas may reuse the benchmark's underlying source pool with independent
named sampling seeds; they are not independent new datasets.

| Role | Grids | Distinct source digits |
|---|---:|---:|
| Nominal pretraining | 2,300 | 20,700 |
| Nominal readiness validation | 200 | 1,800 |
| Initial Fisher and score-noise calibration | 256 | 2,304 |
| Bounded capability/numerical diagnostics | 128 | 1,152 |
| Complete incoming stream | 3,712 | 33,408 |
| **Reserved total** | **6,596** | **59,364** |

Sample source IDs uniformly without replacement and retain all ten digits.
The resulting empirical class marginals are not artificial class balancing,
and finite-population draws are not exactly independent. Record this
approximation in observer diagnostics. Never silently exceed the 60,000-source
pool or reuse a retired training source as a new observation within a replica.
Observer-noise calibration may share the initial Fisher role but adds no
online observation or learner update. Nominal concentration initialization is
$q_0^E=q_0^{\rm ref}=1/2300$, not $1/256$ or $1/(9\cdot2300)$; it remains a
declared influence proxy rather than measured learner covariance.

Preserve the inherited 18-pixel MNIST preprocessing and physical landmark
coordinates, then pad each cell to 40 pixels before transforming it. Assemble
the two-channel reference/transformed grid at $120\times120$. Replace only
the final cell-aligned pooling window, from 14 to 20 after the first factor-two
pool, retaining $6\times3\times3$ features and the $54\to54$ head. Keep SiLU,
average pooling, all trainable base weights, and head-rank-one STM with random
$A$/zero $B$ and $\gamma=1$.

The inherited conservative source-support corner radius 9.5, nuisance bounds,
and one extra interpolation-pixel margin give
$9.5(1.1)\sqrt{1.1^2+1}+2+1<18.54$ pixels, inside the 19.5-pixel half-width.
Test this against the actual preprocessing and interpolation convention; do
not retain the old 28-pixel rejection sampler, which would change the angle
law. Preserve forward transform/label conventions and render inverse sampling
maps correctly. Inspect examples at the support extremes, including corners,
not only typical digits. If a rendering defect invalidates the bound, correct
the common canvas/pooling construction before calibration and record it.

Initialize on nominal pairs with Adam, learning rate .001, batch size 32,
45 epochs, and validation checkpoints at 15, 30, and 45. Freeze the lowest
nominal-validation-loss checkpoint. One common extension to 60 epochs is
allowed before calibration if development validation improves by over 2%
between epochs 30 and 45 and the reserved resource forecast fits. Freeze that
decision for fresh replicas. Do not select initialization using novel test
performance. The reduced pretraining count versus Plan 15 preserves disjoint
sources for the longer stream; report this task change explicitly.

Use all 256 reference grids for the initial integrated-label diagonal; retain
the nested 128-grid estimate only as a stability diagnostic. No dense initial
Fisher is needed. The initial model, FIFO-empty state, diagonal, parameter
ordering, and observer calibration are shared within a replica.

### Independent Holdouts And Clocks

Before development, split official test sources into disjoint 5,000-image
development and confirmation pools. In each replica draw three disjoint
128-grid panels from its assigned pool, with no training-role overlap:

| Panel | Fixed or changing law | Purpose |
|---|---|---|
| Old | Nominal angle law | Retention and calibration |
| Novel | Novel law at independent frozen severities $h\sim U[.5,1]$ | Acquisition and sample equivalence |
| Current | The declared mixture and rotation law at each batch | Tracking through quiet/ramp/return regimes |

Current panels reuse frozen source IDs and base transform uniforms across
checkpoints; the declared law changes, not their independence from training.
No holdout contributes gradients, Fisher observations, observer updates, or
replay. Development results may select recipes; fresh results may not.

Evaluate all three panels initially, every 16 batches, and at exact segment
boundaries. Also evaluate old/novel panels after each batch containing novel
grids until it brackets count 100. Cache duplicate evaluations. A scalar
metric row records both **all incoming grids** and **actual novel grids seen**.
The current-law frozen-model reference varies with the test law and is not a
horizontal line. Fixed-panel frozen references are horizontal.

For the novel-count learning curve, retain the pre-first-novel-batch state at
$n=0$ and each subsequent novel-containing batch's post-solve state, indexed by
its actual cumulative novel count. Intervening nominal training and its cost
are included before the next such state. Interpolate squared risk and
cumulative learning time only between these observed points, onto $n=0,\ldots,100$.
At most four novel grids can arrive in one batch, so a missing integer is at
most three grids from an observed endpoint. This is measurement interpolation,
not a claim that an unevaluated intermediate model existed. Never split batches,
alter $m_t$, or advance the filter to obtain a prettier sample axis.

Do not let subsequent nominal-only training overwrite an already-recorded
novel-count point. Full-trajectory retention and tracking use the all-arrival
clock instead. Show both count axes explicitly. Evicted-training-cohort checks
are separately labeled in-sample diagnostics, not independent test performance.

### Development-Only Retention Calibration

Calibration uses both novel and old development outcomes and exactly two
development replicas, with opposite global sign flips. Fresh confirmation
does not select hyperparameters. Phase 2 below specifies the bounded search.
Keep all attempted recipes and failures in compact artifact tables.

Let $A_j^{\rm new}=100^{-1}\sum_{n=1}^{100}R_j^{\rm new}(n)$ be novel
budget-averaged squared risk on the fixed novel holdout. For retention define
two pixel-valued deteriorations, taking roots within replicas:

$$
H_j^{\rm avg}=\sqrt{100^{-1}\sum_{n=1}^{100}R_j^{\rm old}(n)}
-\sqrt{R^{\rm old}_{\rm init}},\qquad
H_j^{100}=\sqrt{R_j^{\rm old}(100)}-\sqrt{R^{\rm old}_{\rm init}}.
$$

Select B's incumbent with a development deterioration ceiling of .50 pixels
for each mean retention statistic, following the inherited convention. This
is a calibration preference, not a scientific win gate. If no B recipe meets
it, retain the least-violating finite recipe and disclose that result.

Freeze B's selected development mean pair as the reference. For every other
arm, a matched recipe must be within **.10 pixels in either direction** of B
on both means. Prefer the smallest $A_j^{\rm new}$ among matching recipes.
If none match, minimize
$v_j=\max_{\ell\in\{\rm avg,100\}}
(|\bar H_j^\ell-\bar H_B^\ell|-.10)_+$, then novel risk, and label it
unmatched. Do not weaken the tolerance or define matching by nonsignificance.
Better retention outside the band is visible as a different trade-off, not
quietly called matched.

Break numerical ties within $10^{-8}$ of the compared statistic by lower
measured learning time, then lower ridge, lower FD strength, lower process
scale, and stable recipe ID. Full-path final retention is reported separately;
it must not replace retention at the data budget being compared.

Per-arm calibration estimates the performance of retention-calibrated policies,
not a pure intervention with every coefficient held constant. A few shared-
parameter development contrasts can distinguish this from the underlying
algorithm effect. Additional confirmation arms require an explicit scope
revision, not silent expansion.

## Metrics And Interpretation

Primary predictive outcomes are landmark RMS error, squared error per landmark,
and fraction within **one pixel**, separately for new and old
holdouts. Retain Gaussian NLL as a secondary consistent-likelihood measure.
These evaluation losses contain no EWC, ridge, or STM regularization terms.
Report budget-averaged and endpoint outcomes, paired differences, and nominal
95% intervals across complete replica trajectories. Grids, their cells, and
their checkpoints are not independent statistical replicas.

Keep the existing simple efficiency definition. Let A be the common Replay/EWC
reference, $N=100$ novel grids, and $q=\bar R_A^{\rm new}(N)$,
where $R^{\rm new}$ is held-out mean squared error per landmark.
Let $n_j$ be arm $j$'s last interpolated downward crossing of $q$ that persists
through $N$, within observed support. Define

$$
s_j=1-\frac{n_j}{N},\qquad
r_{T,j}=\frac{\bar T_j(n_j)}{\bar T_A(N)},\qquad
\boxed{S_j=\frac{s_j}{r_{T,j}}.}
$$

**Elevator pitch:** $S$ is the fraction of new samples saved, divided by the
compute multiplier needed to reach the same predictive quality. Always show
its two components beside it. Also compare E with B explicitly, labeling the
different reference; do not mix the two targets in one ranking.

Use the same novel test risk and full-likelihood output normalization across
arms. Count unique novel grids and constituent digits separately. Include
replay, FD, transport, observer, and optimizer time in learning compute;
report initialization, calibration, and evaluation costs separately. Check
retention at both full and matched budgets before calling a saving
retention-matched. Reusing the formula preserves meaning, not numerical
comparability to a different historical task or Fisher representation.

Reject an already-achieved initial target. Report unresolved crossings and
bootstrap crossing failures, with no extrapolation, substituted zero savings,
or hidden censoring. Resample whole paired replicas and recompute target,
crossings, costs, and retention together. Sample equivalence is conditional
on this stream and schedule; it does not prove that deleting the saved labels
from a different stream preserves its performance.

Check both frozen initialization and the pre-first-novel state: a target
already achieved without novel exposure is not evidence of novel-sample
savings. Label a reference improvement below 5% from its zero-novel risk as a
weak target, without changing the target. Use 2,000 named-seed paired bootstrap
draws; if any draw has an unsupported crossing, publish its frequency and
withhold an ordinary two-sided inverse interval, while retaining valid
full-budget effect intervals. Do not condition an interval on successful
crossings alone. The reference's saving is zero by definition.

At full and matched novel budgets, report the old-error gap from A (or B for
the explicitly secondary comparison). A .10-pixel one-sided upper tolerance
qualifies a saving as retention-supported when the paired nominal 95% upper
bound is below that tolerance. Otherwise report the saving descriptively
with the retention gap and uncertainty; nonsignificance is not equivalence.
Better retention is allowed at confirmation even though development sought
two-sided matching.

On the all-arrival clock, use trapezoidal integration of squared current-law
error divided by 3,712, then take the square root within each replica. Report
whole-path, ramp, and quiet summaries, plus final old/novel error and one-pixel
accuracy. Do not invert this changing-law tracking curve into samples saved.

Supporting diagnostics remain small: applied/raw $\pi$, movement-to-noise
ratio, both composition states, LTM/STM displacement, age-binned STM
sensitivity, finite STM-ablation predictive KL, eviction-time quality and
subsequent cohort deterioration, transport/projection magnitude, and measured
cost. At a few frozen development checkpoints, compare the archive quadratic
with actual bounded-displacement predictive KL. No dense-Fisher benchmark or
large mechanism factorial is required for this study.

Bound the evicted-cohort check at eight grids per replica: the first four novel
evictees and first four nominal evictees after the initial hold. Record their
predictions at eviction and evaluate at delays of 32 and 128 batches when
within the trajectory. Store source metadata and 54-dimensional predictions,
not a second replay buffer. These inputs are accessible only to the separate
evaluation path, never to training, transport, or the observer after eviction.
At these evaluations, STM ablation means muting the additive adapter at fixed
LTM weights, not merging it into LTM. Label the resulting KL as a diagnostic.

At every 16th batch, a bounded diagnostic may evaluate normalized STM
sensitivity on the two sampled replay grids even for A. Charge that reporting
overhead separately from learning compute, using the same measurement convention
for all arms. At the four development audit states below, compare predictive
KL along $.25u,.5u,u$ with the corresponding archive curvature quadratic on
the fixed replay inputs. This is a local curvature diagnostic, not the full
change in the EWC objective when its center differs from the current model,
and it cannot by itself separate representation error from input mismatch.

The development response audit should use a few cloned states with paired
incoming/replay/probe samples and several fixed $\pi$ values. Examine actual
prediction movement, old/new errors, and the agreement with a scalar-gain
description, alongside the inherited innovation/noise diagnostics. This audit
does not estimate $C$, require an oracle moving parameter, or certify that
learner errors are independent. It tests useful transfer and identifies model
mismatch. Branch results never steer a confirmation trajectory. Failure of the
approximation is a reported scientific result, not permission to add an online
cross-moment estimator or quietly claim the proxy is an exact optimum.

## Phase 0: Implementation And Numerical Acceptance

**Purpose:** build the combined algorithm without importing dense-Fisher costs,
changing the likelihood normalization, or confusing learner state, archive
state, and observer state. This is the numerically delicate phase. Sol should
implement these contracts explicitly before running a development search.

### Package And Data Contracts

Create a separately versioned package, proposed as
`mnist_experiment/combined_memory/`, with entry point
`python -m mnist_experiment.combined_memory.run` and run roots
`cache/mnist_experiment/plan18/production_v1/`. Supply small CPU/CUDA smoke
configurations and a report-only command. Preserve Plans 15--17's sources and
completed artifacts. Record hashes of every reused scientific dependency.
Add a local maintenance guide describing transaction order and protected formulas.

Reuse the grid parameter-layout, Gaussian likelihood, and FD helpers only after
adapting their hard-coded input dimensions in the new package. In particular,
the old grid reference-Fisher and EWC helpers materialize square matrices.
Do not call them and then discard off-diagonal entries. The production archive,
including initialization, must contain only 3,268 diagonal entries.

Record a fixed parameter order, LTM/STM indices, LoRA scaling, target ordering
(cell, landmark, horizontal/vertical displacement), and the 3,376 total count.
Functional parameter replacement must include all STM coordinates when
evaluating either endpoint. Never merge adapters into the training model,
rebalance their factors, or change parameter charts during a trajectory.
The observer uses a separate merged predictor at its fixed initialization
anchor; assert its predictions equal the unmerged model at that anchor.
Map merged coordinates explicitly to the LTM ordering, including names and
offsets, before applying the archive diagonal as a risk metric.

Use two CPU threads initially, deterministic seeded PyTorch operations, no AMP,
dropout, TF32, stochastic augmentation inside solves, or random global-seed
coupling. Initialization, source allocation, transforms, mixture flags, replay,
FD probes, diagnostics, and bootstrap receive separately named component seeds.
Render deterministically on CPU and move bounded batches to CUDA; do not create
a disk-offloading experiment or cache all rendered trajectories.

### Exact Gaussian Quantities And Diagonal Accumulation

For one 54-output grid, implement and test

$$
L=\|f-y\|^2/8+27\log(8\pi),\qquad
s=J^T(y-f)/4,\qquad
z_{Q,L,i}=\frac1{4|Q|}\sum_{x\in Q}\sum_{o=1}^{54}J_{x,o,i}^2.
$$

Average over observations, not coordinates or output chunks. Compute per-grid
observed scores for the filter, never the outer product of a batch-mean score.
Fisher rows integrate Gaussian labels and do not depend on observed residuals.
Accumulate vector diagonals directly from input/output-Jacobian chunks.
Start with two inputs and six outputs per chunk; reduce either for memory or
stop latency, preserving exact sums. All 54 outputs must contribute once.

Keep model training in float32 initially, but evaluate Fisher Jacobians,
observed-score passes, endpoint subtraction, archive operations, and the filter
in float64. Casting already computed float32 Jacobians to float64 is not a
float64 derivative pass. Use functional model evaluation with float64 inputs
and parameter copies, leaving accepted training state untouched.
Do not materialize full per-sample Hessians, an LFU Jacobian, or a dense Fisher.
Tiny hand-check matrices in unit tests are not experimental Fisher estimates.

For paired transport, use exactly matched input IDs, transforms, weights, and
output chunks at both accepted endpoints. A stable equivalent accumulation is

$$
\Delta_{t,i}=\frac1{4|Q_t|}\sum_{x,o}
(J^+_{x,o,i}-J^-_{x,o,i})(J^+_{x,o,i}+J^-_{x,o,i}).
$$

This avoids subtracting two large accumulated sums of squares. Also retain
$z^+$ for the eviction blend. Check parity with direct endpoint subtraction
in float64. **Do not divide by the displacement norm or by an FD stencil
step:** the auxiliary correction is a finite change, not a derivative estimate.
If the accepted displacement is exactly zero, the correction must be zero.

Calculate the full unprojected archive blend before clipping its diagonal.
Use `-expm1((e/4)*log1p(-pi))` for retirement weight, and explicitly handle
$e=0$. Log the negative-entry fraction, negative mass relative to old trace,
and projection magnitude before clipping. Do not clip the correction itself.
There is no eigendecomposition, low-rank compression, transport damping,
correction-size cap, or omitted residual term in the primary method.

### FD KL, Optimizer, And Numerical Floors

Use two named-seed Rademacher probes of dimension 108 and $h=.01$, embedded
only in STM. For the fixed Gaussian likelihood the symmetric FD quantity is
$\|f_+-f_-\|^2/(4h^2)$. Differentiate both branches through the candidate
parameters; do not detach a teacher or replace this with directional KL.
Average probes, divide by 108, and apply the frozen squared replay ages.

For this single bilinear head, the symmetric prediction difference cancels
the quadratic-in-$h$ term exactly in real arithmetic. JVP agreement is
therefore a particularly strong value/gradient test. JVP is a test oracle,
not another experimental condition. Check $h=.005,.01,.02$ on fixed inputs:
unexplained stencil dependence signals rounding or an implementation error.
If float32 prediction subtraction is inadequate, evaluate both FD branches
in float64 while preserving gradients back to the optimizer variable.
Do not increase $h$ or rescale the penalty to conceal a failed identity.

Use a fresh L-BFGS solve per incoming batch, strong-Wolfe line search, learning
rate 1, history size 20, at most 100 outer iterations and 150 closure calls,
physical-gradient infinity tolerance $10^{-7}$, and objective-change tolerance
$10^{-9}$. Fix the data, replay sample, probes, archive, anchors, ages,
$\pi$, $\kappa$, and numerical scaling for the entire solve. A new closure
is not a new observation, Fisher measurement, or probe draw.

To avoid the poor conditioning observed in earlier plans, solve in fixed
diagonally scaled coordinates $\theta=\theta_t+a\odot z$, initially $z=0$.
With $\epsilon_s=\max(10^{-12},10^{-8}s_0)$, use

$$
a_{L,i}=[d_{t,i}+\omega_t(d_{t,i}+\kappa_t)+\epsilon_s]^{-1/2},
\qquad
a_{S,i}=[s_t+\omega_t\kappa_t+\epsilon_s]^{-1/2}.
$$

Clip these numerical scales to $[10^{-4},10^4]$, recording the fractions.
This invertible change of solver coordinates does not add a regularizer or
post-scale the accepted update. Recover the physical gradient by dividing
the coordinate gradient by $a$ before testing convergence. Audit finite-budget
behavior; an equivalent optimum does not imply equivalent truncated solves.
Do not use the observer's diagonal floor to inflate the EWC archive.

Evaluate the initial point and retain the lowest finite, fully evaluated
objective/gradient pair encountered. Never accept the optimizer's unevaluated
last trial. Reset L-BFGS history whenever rewinding to a different evaluated
point. Record solver limits, physical residual gradient, objective decrease,
rejected/nonfinite trials, and whether a better point than initialization
was found. Never multiply the final displacement by $\pi$.

An invalid initial objective or gradient is a failed unit, not a successful
no-op. A recoverable failed trial may return the finite evaluated best with
an explicit numerical flag. Repeated failures must trigger a documented
numerical diagnosis, not silent replacement of an entire condition by no-ops.
During development, one common increase to 200 iterations/300 calls is
permitted if the doubled-budget audit below shows material undersolving.
Changing precision or solve limits applies to all subsequently compared arms,
with prior development retained and incompatible versions not pooled.

### Kalman Implementation Contract

Use the fixed initialization anchor for the entire trajectory, with no automatic
reanchoring. Measure incoming observed-label scores in merged LTM-sized
coordinates, never raw joint coordinates or a zero-mean model-label score.
The anchor mapping is
$\max(d_{a,i},\max(10^{-12},10^{-6}\operatorname{mean}d_a))$ coordinatewise.
Its floor is for the observer only. Retain and report zero/weak coordinates
rather than silently selecting a favorable subspace.

Compute initial observed-score variance from the 256 calibration grids using
centered sample moments with $ddof=1$. Let $K_{0,i}$ be that variance divided
by the squared mapping. Initialize state means at zero and both diagonal
state variances at $K_{0,i}/2300$, with zero position-velocity covariance.
Use per-coordinate $2\times2$ states, not one dense joint covariance.

For every incoming batch use

$$
F=\begin{pmatrix}1&1\\0&1\end{pmatrix},\qquad
H=(1,0),\qquad
Q_i=\eta_{\rm KF}K_{0,i}
\begin{pmatrix}1/3&1/2\\1/2&1\end{pmatrix}.
$$

$K_0$ and the anchor mapping are frozen. Measurement-noise variance is the
pre-batch score-variance EMA divided by the squared mapping and batch size
four. Compute its gain from the prediction covariance and use the Joseph
posterior update
$(I-KH)P^-(I-KH)^T+KRK^T$ in float64, then symmetrize. Update the score-variance
EMA **after** assimilation with weight $1-2^{-1/32}$, not before scoring that
same innovation. Preserve the inherited finite noise floor
$\max(10^{-30},10^{-12}\operatorname{mean}R)$ and log its use.

Roundoff-scale negative covariance diagonals or excessive cross terms may use
the existing explicit $2\times2$ PSD repair, with pre-repair diagnostics.
Do not use covariance repair to hide nonfinite states or substantive invalid
covariances. Check innovations, covariance repair frequency, and mean-motion
versus uncertainty contributions separately.

Choose the action before reading the new batch using the **pre-transaction
archive diagonal**, the previous filter state, and $q^{\rm ref}$. Stabilize
the ratio by dividing movement and noise terms by their common finite maximum
before forming numerator/denominator. Retain the 16-batch .025 cold start,
$[10^{-4},.2]$ bounds, and bounded covariance-only fallback
$4q^{\rm ref}/(1+4q^{\rm ref})$ for unsupported finite estimates. A corrupted
nonfinite state is an error requiring recovery, not indefinite fallback.
Record every fallback, clip, and raw recommendation.

The state prediction/assimilation occurs once after the batch and is eligible
only for the next action. Update $q^{\rm ref}$ once with the applied action
and four fresh grids; update $q^E$ only on eviction. Save both. The field
$\eta_{\rm KF}$ is process-noise scale, not retirement weight $\eta_t$.
There is no $C$ state or estimated cross-moment term in the controller.

### Atomic Transaction And Smoke Acceptance

One production transaction is: read the accepted checkpoint; choose action;
materialize the four arrivals; stage FIFO and uniformly sample two distinct
old residents (or all residents when fewer); freeze ages/probes; solve;
measure paired transport and evictee diagonal; blend/project and recenter
only on eviction; assimilate the fresh anchor scores; update counters;
publish the next checkpoint. Actions never see labels, mixture flags, or
future score tape before their decision.
Use zero-based internal step indices and one-based displayed batch numbers;
the cold start therefore covers internal steps 0--15, displayed batches 1--16.

Resident age is squared clipped elapsed incoming-grid count divided by 32,
using the existing insertion-count convention and pre-solve staged count.
Evictees remain eligible for that batch's replay until commit. Newly arriving
grids are not also replayed in the same transaction. Save full pre/post
parameter vectors for transport recovery, and freeze the initial Fisher
evidence and all online consolidated-ID ownership separately.

Verify all of the following before development:

- Renderer/labels round-trip; support extremes; nonblank images; fixed parameter
  count; same-chart merged prediction and score parity.
- Gaussian score sign, summation over 54 coordinates, diagonal chunk invariance,
  and FD values and gradients against analytic/JVP checks. Small smooth CPU
  checks should agree to relative $10^{-6}$ where the reference is nonzero;
  use an explicit $10^{-10}$ absolute tolerance near zero.
- Paired transport under joint STM/LTM movement, zero move, and fixed-feature
  head-only movement; direct difference versus product accumulation and a
  small-step directional derivative check. No production dense Fisher call.
- Archive-before-action timing, one update per new batch, no early evidence
  admission, no anchor recentering without eviction, and no double retirement.
- Filter formulas against Plan 17 on identical inputs, cold start, saturated
  finite signals, unsupported-estimate fallback, Joseph PSD behavior, and
  restoration of both composition states.
- A tiny two-replica five-arm CPU run, then CUDA parity/recovery smoke with
  short regimes, smaller buffers and solver limits but the real model. Use
  separate smoke identities, not shortened production trajectories.
- Interrupt before publication, during checkpoint writing, and after publication
  before metric export. Resume must reproduce accepted state and avoid duplicate
  observer or archive updates. Completed-unit resume must perform no training.

Create the reporting notebook and restart notes during this phase. The notebook
must clearly show pending results rather than running training to fill them.

## Phase 1: Readiness, Numerical Audit, And Timing

Initialize the two development replicas and run A at ridge ratio 1 through
the complete schedule on each. These are reusable calibration paths, not
extra confirmation replicas. Instrument fresh fixed-anchor scores for shadow
filter calibration; this is development-only overhead, not part of A's
production algorithm or its standalone learning-time comparison.

Show initial old/novel errors and constant-zero-displacement references. On
the first development replica only, permit one capability fit using 96
diagnostic-role grids at fixed novel severities and the other 32 for validation,
at most 10 epochs of Adam .001. It never initializes a scientific path.
Poor capability is reported; it does not authorize widening the architecture,
changing the likelihood, or switching tasks to rescue a treatment.

Save pre-solve checkpoints at batches **32, 312, 464, and 616** on the first
A path. They cover aged replay during the initial hold, mid-outbound, peak,
and mid-return states. These four states are the bounded numerical/audit set:

1. Check FD values and gradients at three stencil sizes and float32 versus
   float64 prediction arithmetic, with the same two probes and replay ages.
   Require relative value/gradient disagreement no more than 1% where resolved;
   use the float64 path for unresolved cancellation, not a smaller denominator.
2. Compare paired Jacobian-product transport to float64 direct differences,
   with absolute error scaled to endpoint magnitude when the true difference
   is near zero. Never demand a misleading relative tolerance on a zero update.
3. Compare the ordinary solve to doubled budgets at the same objective,
   action, inputs, and probes. Flag normalized objective improvement above
   $10^{-3}$ or more than 5% additional prediction movement on the fixed
   diagnostic inputs. Remove the Gaussian additive constant when normalizing
   objective improvement. Apply the common numerical repair rule before tuning.
4. Run cloned one-step actions at $\pi\in\{.01,.025,.1\}$ on those states,
   holding every other quantity fixed. Show prediction movement, old/new
   development error, physical residual gradient, and scalar-gain mismatch.
   These at most twelve solves are a diagnostic, not an oracle policy or
   a new fixed-action search.

From the same four states compute a common FD strength scale
$\lambda_0=.1\,\operatorname{median}\|g_{\rm data}\|_2/
\operatorname{median}\|g_{\mathcal C}\|_2$ using the normalized age penalty
and all trainable coordinates. Exclude only nonfinite/zero-denominator
measurements, with reasons; require at least two resolved states. Otherwise
diagnose routing/scaling before running an FD search. This is a gradient-scale
initialization, not a claim that a particular norm ratio is optimal.

Audit the fixed-anchor mapping at these four states using at most 32
already-available training grids per state; compare scores at the initialization
anchor and the current merged predictor. Report discrepancies relative to
score variability without fitting a population optimum or estimating $C$.
Historical samples used for diagnostics never enter online filter updates.

Measure complete-path learning, observed-score/Jacobian work, FD, solver,
evaluation, initialization, checkpoint/report time, peak RAM/VRAM and bytes.
Use explicit CUDA synchronization at timing boundaries. Add one bounded
128-batch C and D timing probe from the same saved development state if their
extra overhead cannot be measured from the audits; these probes do not select
recipes or masquerade as full paths. Predict the remainder with 25% contingency
and reserve at least one active hour for closeout.

Proceed to calibration with finite supported settings regardless of effect
sign. Numerical invalidity requires repair and revalidation. An unaffordable
forecast requires the bounded cohort/resource rules below, not shorter regimes
or under-solved treatments.

## Phase 2: Bounded Retention Calibration And Design Freeze

Use ridge ratios $c_{\rm ridge}\in\{.1,1,10\}$ and FD multipliers
$\lambda_F/\lambda_0\in\{.1,1,10\}$. All candidate runs cover the full 928
batches, even though the selection acquisition metric is the first-100-novel
budget average. No short-path screening that skips the return or quiet interval.

The order limits compute without making the user a selection bottleneck:

1. On development replica 1, run B at ridge 1 for all three FD multipliers.
   Rank by the .50-pixel retention ceiling, then novel risk, using positive
   ceiling violation first if none are feasible. For the winning strength,
   add ridge .1 and 10: five B recipes in total, not a full nine-point product.
   Run the top two of these recipes on replica 2 and select B using paired
   two-replica means under the same rule. Freeze its FD strength for C, D, E
   and its retention means as the matching reference.
2. For A and C, run all three ridge candidates on replica 1, comparing against
   B's corresponding replica-1 retention. Continue the top two per arm on
   replica 2, then select using the two-replica matching rule. Reuse readiness
   A/ridge-1 paths where identical.
3. Replay the already recorded fresh-score tape through shadow velocity filters
   at $\eta_{\rm KF}\in\{10^{-5},10^{-4},10^{-3}\}$. Rank finite candidates by
   mean predictive innovation log score, equally weighting the two replicas
   and four regimes. Choose the top two; ties prefer the smaller scale.
   Shadow processing is development analysis, not live look-ahead. Score
   predictability is not proof of tracking the true network optimum.
4. For each shortlisted process scale, run D and E at all three ridges on
   replica 1. Continue the best retention-matching ridge for each
   arm/scale pair on replica 2. Choose **one common process scale for D and E**
   by smallest maximum two-arm retention mismatch violation, then mean
   two-arm novel risk and the declared tie-breaks. Freeze the corresponding
   per-arm ridge choices. This preserves a common observer rather than giving
   each LFU setting an unrelated filter.

This is at most **34 complete development trajectories including the two
readiness paths**: seven B paths, five each for A/C, sixteen for D/E, and
at most one otherwise unused readiness A path on replica 2. Cache compatible
completed candidates by identity, not by rerunning them. Preserve the
single-replica screen and two-replica finalists as distinct evidence.

Allow at most **two additional ridge recipes**, each run on both development
replicas, if an arm remains unmatched or B still violates its .50-pixel
ceiling. For B, prioritize its positive ceiling violation; for other arms use
the declared matching violation. Choose a geometric midpoint of the
nearest explored bracket, or one adjacent decade beyond a selected boundary;
prefer the largest mismatch, then stable arm ID. Zero ridge is eligible as
the lower endpoint extension when the .1 boundary remains overprotective.
Inspect the acquisition/retention plot before proposing the recipe: do not
assume monotonicity from three noisy points. Record the reason and use the same
selection rule afterward. No FD/process grid expansion or per-coordinate gains.
Omit this optional four-path work first if the resource forecast is tight.

If optional refinement changes B, its strength stays fixed and the reference
retention means change transparently; re-rank the already evaluated controls
against that reference without extra hidden searches. The final common process
scale must still be selected jointly for D/E. Unmatched methods continue as
explicit trade-offs, not mislabeled retention-matched comparisons.

Before fresh outcomes are read, seal the numerical policy, source identities,
renderer/schedule, initial-fit rule, FD strength, five ridges, shared process
scale, all candidates/reasons, holdouts, metric definitions, seeds, cohort size,
and cost/storage forecast. Freeze even an unfavorable selection. Do not tune
on the confirmation notebook or declare a phase successful by a p-value.

## Phase 3: Fresh Paired Exploration

Target **six fresh changing replicas and two fresh stationary replicas**,
all with five conditions: **40 full trajectories**. Changing replicas balance
global sign flips three per sign; stationary replicas use opposite signs only
to define their untrained novel panel. Use independent initialization and stream
seeds, with complete within-replica pairing. Do not reuse development paths as
confirmation or treat nine cells as nine independent replicas.

Before confirmation starts, use measured complete-path costs plus 25%
contingency and reserved closeout time to choose the affordable allocation.
The predefined smaller allocation is **four changing plus two stationary
replicas**, 30 trajectories. Prefer six changing replicas when affordable;
otherwise choose four. Do not drop a condition, change a reference, or shorten
the stream. If the smaller allocation plus remaining calibration cannot fit,
report the resource constraint and the completed development evidence rather
than imply the full experiment finished or exceed 20 hours.

Interleave complete five-condition blocks and rotate condition order by a
pre-generated seed to limit warmup/time-of-day bias. Insert one stationary
block after the first two changing blocks and one before the last changing
block. Only complete blocks enter paired inferential summaries; partial blocks
remain visibly resumable progress. The frozen allocation is not expanded or
stopped because intermediate intervals appear promising.

Publish E-B and E-A as the main comparisons, C-B and D-B as incremental
comparisons, and the conditional interaction on the lower-is-better novel-risk
scale. Report stationary comparisons separately. Use shared whole-replica
bootstrap draws across all contrasts. With only two stationary replicas, show
both paired trajectories and descriptive differences rather than a misleading
95% bootstrap interval; label the stationary cohort a diagnostic.
For changing replicas, stratify the paired bootstrap by global sign flip,
resampling the three (or two) replicas within each sign together across arms.

Learning time is measured per standalone arm: adaptive arms include their
own fresh-score/filter work and LFU arms their paired endpoint work. Shared
initial assets are disclosed separately. Do not credit a method with zero
observer cost because another condition computed a compatible tape earlier.
Warmup/profiling overhead and the study's actual total runtime are separately
recorded. Equal-data and matched-quality costs are different columns.

## Phase 4: Human-Readable Report And Closeout

Keep the main notebook to roughly **five multi-panel figures and three compact
results tables**, plus a small progress indicator, brief motivation, definitions,
qualifications, and conclusion.
Detailed candidates, per-coordinate diagnostics, logs, and proofs are linked
artifacts or plan sections, not pages of printed tables.

1. Near-top progress: phase/revision, completed/planned full paths and paired
   blocks by development/changing/stationary role, active batch, heartbeat,
   charged hours, forecast remainder, and disk headroom.
2. Task/readiness figure: rendered grids, rotation densities, regime envelope,
   expected/actual novel counts, and frozen predictive references. Explicitly
   label incoming grids, novel grids, and constituent digit pairs.
3. Development figure: acquisition versus retention with selected recipes,
   plus compact numerical/observer health diagnostics. Settings table includes
   all five arms, parameter counts, ridge/FD/process settings and match status.
4. Fresh predictive figure: fixed novel/old learning curves on novel-count
   axes, and current-law/quiet/return behavior on the all-arrival clock.
   Include one-pixel accuracy, RMS units, uncertainty, and labeled frozen lines.
5. Mechanism/health figure: mean applied weights and movement/noise, archive
   correction/projection, and a concise STM-sensitivity/eviction diagnostic.
   Do not infer information transfer from these proxies alone.
6. Efficiency figure and conclusion tables: observed common-target crossings,
   percent samples saved, matched-quality and equal-budget compute, $S$,
   paired effect intervals, retention qualification, and unresolved inversions.

The numbered items specify content, not six mandatory figures: progress is a
small table, and the remaining five items each have at most one multi-panel
figure. Use one settings table and two compact conclusion tables. Add sections
chronologically as phases complete; no newest-first rearrangement.

Explain sample equivalence in one sentence beside its plot: fewer novel grids
to the same independent-test error, at the reported retention and compute
cost, under this stream. Include interpolation resolution and the 152 expected
full-trajectory novel grids versus the 100-grid analysis budget.

Publish atomically at phase/block boundaries, graceful pauses, and at most
every 15 active minutes, with a lightweight heartbeat between reports.
Use generation hashes to skip unchanged report work on resume. The notebook
never trains, downloads, repairs artifacts, or calculates large matrices.
Inspect plots visually before closeout for legends, units, confidence bands,
regime placement, and consistency of count axes.

Close with what the combined method contributes beyond B, what it contributes
beyond A, retention/cost limitations, and whether the observer approximation
transferred usefully. State that replay-aware error correlation and the exact
optimality of the combined learner remain unproved. Preserve all revisions,
unfavorable development results, and resource-limited partial outcomes.

## Resources, Recovery, And Autonomy

Initial programming is provisionally **4--8 engineering hours**, separate from
active experimental compute. Computational tests are charged below. Historical
grid paths took roughly 447--548 seconds under different Fisher/rendering
settings; the longer padded stream and added observer/transport work make this
only a rough reference, not a transferable throughput measurement.

| Phase | Provisional active compute | Planning allocation |
|---|---:|---:|
| 0: implementation acceptance | .5--1.5 h | 1.5 h |
| 1: readiness/audits/timing | 1.5--3 h | 2 h |
| 2: bounded calibration | 4--7 h | 5 h |
| 3: fresh paired exploration | 5--9 h | 9 h |
| 4: report and closeout | .5--1 h | 1 h |
| Repair/restart reserve | As needed | 1.5 h |
| **Total planning range / ceiling** | **11.5--21.5 h before reserve** | **20 h maximum** |

The upper forecast does not fit automatically. Plan practically for roughly
14--20 active hours, replace estimates with measurements in Phase 1, and enforce
the common ceiling. Phase allocations are planning envelopes, not independent
allowances: mandatory work may move between them within the same 20 hours.
Do not launch optional refinements without reserving the smaller complete fresh
cohort and at least one hour for closeout. If required work becomes unaffordable,
preserve/report it and stop at the compute limit rather than promise completion.

Each accepted batch is a recoverable transaction. `--resume` must restore model,
pre/post transport references, FIFO contents and ages, unique-ID ledger,
observer state, composition states, RNG/component seeds, and budget accounting.
Temporary outputs never count as complete. A sealed `COMPLETED` unit is
append-only, with configuration/source/schema hashes and environment provenance.
Record mathematical repairs under new source/protocol identities; do not mix
incompatible trajectories in one contrast.

Provide `--resume`, `--request-stop`, `--clear-stop`, `--max-units`,
and a cumulative `--max-active-hours` limit, with a single-writer lease and
stale-PID recovery. The intended eventual invocation, once implemented and
authorized, is:

~~~bash
~/.venv/bin/python -m mnist_experiment.combined_memory.run \
  --config mnist_experiment/combined_memory/configs/production.json --resume
~~~

Do not require the user to reconstruct state from conversation history.
Checkpoint pretraining at bounded minibatch intervals and online learning after
every accepted arrival. Use temporary files, flush/fsync, atomic publication,
and an accepted transaction ID tying model, archive, FIFO, controller, and
metrics together. A crash before publication replays that one arrival from
the old checkpoint; a crash afterward must not repeat its assimilation or
retirement. Repair derived metric/report exports from accepted records, never
invent accepted scientific state from a partially written file.

Honor graceful stops between derivative chunks, optimizer closures, data
batches, artifact checks, and report chunks. Target release within one minute
and record the longest noninterruptible kernel during acceptance. Hard stops
may lose at most one online arrival or one bounded initialization/analysis
chunk. Check both replay-ID and observer-assimilation counters when resuming.
Source/configuration/dtype mismatches require an explicit compatibility
decision or a new revision, not best-effort silent loading.

Persist active-time accounting and a lightweight heartbeat at least every
15 seconds. Charge failed/retried work, derivative audits, evaluation, I/O,
and reporting; exclude machine downtime, but record uncertainty in the final
interrupted chunk conservatively. Reboots and new invocations do not reset the
20-hour ledger. Do not count overlapping work twice if later implementation
parallelizes CPU preparation with GPU work.

Enforce a **500,000,000,000-byte peak Plan 18 artifact/dependency budget** and
at least **20 GB actual free-disk reserve**, including temporary checkpoints
and finalization copies. Forecast an incremental 5--20 GB initially and replace
that with Phase 1 measurements. Store source IDs, transformations and compact
metrics, not rendered streams, per-step Jacobians, or dense matrices.
Retain two rotating checkpoints for an incomplete trajectory, its initial/final
accepted states, and only the prespecified audit snapshots. Completed units
are immutable; pruning unrelated or completed scientific data is not authorized.

Resume validates source/configuration seals, active checkpoint contents, and
required dependencies. Full completion validation and the final paired audit
remain mandatory, but must be cancellable and must not trigger repeated
notebook reconstruction. The reporting generation records its input seals and
excludes incomplete blocks. Keep an append-only correction/validity log and a
concise `mnist_experiment/PLAN18_HANDOFF.md` with exact restart command,
phase, active unit/batch, completed/remaining work, resource ledger, and any
numerical qualifications.

After execution is authorized, Sol may make documented implementation fixes,
the specified numerical repairs, bounded calibration choices, and the timing-
based cohort reduction without waiting for model switching or routine review.
Preserve every material scientific revision and explain it in the notebook.
Revalidate affected controls together; if a defect invalidates selected recipes,
recalibrate or report the resulting resource limitation rather than pooling
incompatible results.

Changing the likelihood, source/target law, five conditions, common reference,
retention tolerances, sample-efficiency definition, or adding an estimated-$C$
controller requires a new scientific amendment. No automatic next experiment,
efficacy gate, significance stopping, or significance-driven extension.
Exploratory theory checks for the correlated-blend and profiled-precision
identities are welcome as tiny CPU tests; they are not production estimators.

## Interpretation Boundary

The attachment supports a coherent **local** extension: redundant STM
coordinates can change optimization and archive faithfulness; transport can
update the LTM curvature representation; an optimal evidence-weight concept
survives when bias, covariance, and dependence are specified. None of these
statements promises that the three empirical gains compose.

The controller choice is settled: retain Plan 17's calibrated velocity-Kalman
observer and reference-composition rule, with fresh-only observer measurements
and no online estimation of $C$. The remaining scientific uncertainty is how
well that approximation transfers to the combined learner, not whether this
experiment must first solve another cross-covariance estimation problem.
Development-only retention calibration and the bounded response audit assess
that uncertainty without expanding the five conditions or compute allocation.

Schedules, data budgets, candidate sets, retention tolerances, audit states,
and resource contingencies are now specified. Development freezes the allowed
choices before any confirmation outcome is inspected. No change to the
top-level mathematical overview is needed to execute this exploratory study.

## Execution Record And Conclusion

Production root: `cache/mnist_experiment/plan18/production_v1`.
The study completed at `2026-11-07T08:55:55Z`. The full six-changing/two-stationary
allocation was affordable and completed; no timing reduction, optional ridge
refinement, efficacy stopping, or significance-driven extension was used.
Calibration froze all recipes before fresh outcomes were generated.

| Fresh comparison | Novel-risk difference (pixel squared), nominal 95% CI | Old-error difference at 100 novel grids (pixels), nominal 95% CI |
| --- | --- | --- |
| Combined E minus STM incumbent B | 0.317 [-0.089, 0.717] | -0.150 [-0.239, -0.067] |
| Combined E minus Replay/EWC A | 0.220 [-0.223, 0.620] | -0.110 [-0.201, -0.020] |
| Transport C minus incumbent B | 0.140 [0.027, 0.286] | -0.017 [-0.032, -0.002] |
| Adaptation D minus incumbent B | 0.141 [-0.337, 0.544] | -0.084 [-0.174, -0.002] |

Negative differences favor the first arm. Novel risk averages independent-test
MSE over actual novel budgets 1-100; old-error differences compare RMS at the
100-novel-grid checkpoint. Intervals use the prespecified paired sign-stratified
bootstrap and are nominal, not familywise. There is no demonstrated additional
novel-learning benefit from the full combination here. Its lower old error at
the endpoint is useful retention evidence, not proof of extra sample efficiency.
The near-zero whole-path current-law difference also does not establish a
general predictive advantage. Historical gains do not automatically compose.

For STM-only B versus A, the observed common-target crossing is 85.0 versus
100 novel grids: 15.0% sample saving, 1.45 times matched-quality learning compute,
and S=0.103. S is sample-saving fraction divided by the compute multiplier.
These are exploratory point estimates: 14.1% of bootstrap draws have unsupported
crossings, so inverse intervals are withheld, and fresh matched-budget retention
support is not established. C's point saving is 2.85% (S=0.0153); D/E do not reach
the common A target within the observed budget. Do not report these as reliable
retention-controlled sample-saving gains. E costs 1.93 times A at equal 100-grid
budgets and 1.24 times B; those are not matched-quality savings estimates.

The inherited Kalman action settled near a low adaptation weight rather than
reproducing Plan 17's benefit in this combined grid setting. Fixed-anchor score
mapping discrepancy, diagonal-archive approximation, finite solves, and replay
dependence remain qualifications. The experiment does not establish the exact
optimality of the combined action or prove information transfer. Stationary
replicas are retained as descriptive diagnostics, not extra inferential replicas.

The final audit verified all 99 units present before its own completion seal,
all eight paired blocks, full exposure/retirement accounting, and finite
nonnegative archive diagonals. The final CPU suite passed 85 tests with one
CUDA-only skip; explicit CUDA recovery/parity and CLI lifecycle acceptance had
already passed. `mathematical_overview.ipynb` and all sealed scientific sources
are unchanged. Completed units and prior report generations remain immutable.

One reporting-only repair raised the aggregate scalar-JSON input guard from
64 to 256 MiB after the first stationary block. Regression tests and a production
rebuild passed before resumption; `execution_notes.json` records the repair.
No estimator, likelihood, stream, calibration recipe, or analysis changed.
The final notebook contains five figures and three result tables, with explicit
completion status, unsupported-crossing labels, and a qualified conclusion.

The study charged about 7.37 hours through completion and 7.38 hours including
final acceptance/reporting. Fresh execution after selection took about 4.05
active hours, including fits and in-run reporting. The Plan 18 artifact family
uses about 0.65 GB plus 0.067 GB cached MNIST dependencies, well below 500 GB;
actual free storage remained about 577 GB. No experiment process is running,
and no further experiment or cohort expansion is authorized automatically.

## Phase 5: Broader Continuous Drift

**Status:** complete. Execution was authorized on 2026-11-07; the detached run
finished all Phases 5A-5E at 2026-11-08T04:44:45Z (November 7, 8:44 p.m.
Pacific). Post-reboot checks on November 8 passed full artifact verification.
The separately versioned `mnist_experiment.combined_memory.broad_drift.run`
entry point needs no further execution. Completion details are recorded in
`PLAN18_HANDOFF.md`.

### Motivation And Question

The practical objective remains sample-efficient continual learning under
limited replay and compute. STM has produced strong results in earlier grid
experiments, but the present combination has not recovered the incremental
benefits previously observed for LFUs and velocity-Kalman $\pi$.

The supplemental B-A contrast provides useful context: STM reduced whole-path
current-mixture RMS by .0116 pixels, with nominal paired 95% interval
[-.0328, -.0002]. Its novel-risk difference was -.0969 pixels squared
[-.3170, .0206], and its 15% point sample saving remains qualified by unsupported
bootstrap crossings and matched-budget retention. Thus STM retains a small
tracking advantage here; this does not yet establish that it reliably detects
tiny distribution changes or that its novel-data benefit is significant.

Meanwhile, adaptive $\pi$ averaged about .00349. The recorded weight/noise
recursions and an independent score/filter replay revealed no arithmetic defect
in the checked calculations. A more plausible next hypothesis concerns the
process: only 4.23% of fresh grids came from the changing branch, and 85.06%
of batches contained none. The observer sees the full incoming population,
whereas novel-test error concerns the conditional rotated population. Small
population-mean movement and mixture variability can therefore coexist with a
difficult novel-domain task. Fixed-anchor mapping error remains an alternative
explanation, not something that this audit ruled out.

Plan 17's successful continuous-rotation experiment moved the entire incoming
distribution. This extension asks: **when slow drift affects the whole stream,
do transport and adaptive $\pi$ add useful predictive or sample-efficiency
benefits beyond STM, at comparable retention?** Increase the breadth of drift,
not its speed. A larger $\pi$ is a diagnostic outcome, never the success target.
LFUs correct parameter-location staleness rather than detect regimes; stronger
distributional motion may make that correction more useful, but need not do so.

This is one new process, not a factorial search for a favorable task. Do not
increase Kalman gain by hand, weaken the fixed-.025 baseline, add discrepancy
corrections, or change the model/likelihood to obtain a positive result.

### Process And Conditions

Retain the original 928-batch timeline, four fresh grids per batch, 160/304/304/160
quiet/outbound/return/quiet lengths, cosine schedule $h_t$, fixed cell signs,
and all nuisance-transform laws. **Remove the Bernoulli nominal/novel branch.**
Every incoming cell $j$ now has rotation

$$
\alpha_{t,j}\sim\operatorname{TruncNormal}
\bigl(s_j45h_t,6^2;[-70,70]\bigr)\quad\text{degrees}.
$$

Quiet intervals therefore reproduce the nominal law exactly. All cells' means
move coherently according to their fixed signs during each ramp; their sampled
angles and other transforms remain independent conditional on the schedule.
There is no peak hold, abrupt component switch, or increased maximum location
increment: it remains about .233 degrees per incoming batch. The broader
population change could still challenge local approximations; smoothness alone
does not prove a LAN regime. Measure score movement and mapping discrepancy.

Keep the 54-output Gaussian likelihood, 3,268 LTM plus 108 head-rank-one STM
coordinates, strictly diagonal EWC, FIFO32/replay2, initialization protocol,
eviction-only archive admission, actual-displacement paired FD transport,
and predictable fresh-score velocity observer. Keep the five arms:

| ID | Method | STM age FD KL | LFU | $\pi$ |
|---|---|---|---|---|
| A | Replay/EWC | No | No | .025 |
| B | STM incumbent | Yes | No | .025 |
| C | STM + transport | Yes | Yes | .025 |
| D | STM + adaptation | Yes | No | Velocity Kalman |
| E | Combined | Yes | Yes | Velocity Kalman |

A retains the same trainable STM parameters. Preserve the frozen-model
evaluation reference. Fresh stationary diagnostics use $h_t=0$ throughout.
The existing source budget still fits: 3,712 incoming grids / 33,408 digit
pairs, with 59,364 distinct training-source digits across roles per replica.
Use new development/confirmation seed namespaces, preserving the declared
test-source partition and within-replica pairing. Never relabel an earlier
observed confirmation trajectory as fresh evidence for this amendment.

### Evaluation And Sample Clock

Retain three independent 128-grid holdouts: fixed nominal, fixed rotated at
frozen severities $h\sim U[.5,1]$, and current-law. Update only the current-law
panel's generating rule to the whole-stream law. Evaluate all three initially,
every 16 batches, and at boundaries. Also evaluate fixed old/rotated panels
after every outbound batch. Holdouts supply no training, observer, or Fisher
updates; their losses contain no regularizers.

The old **100 rare-grid** budget is not meaningful for this process. Define
$n=4j$ after outbound batch $j$, with **$N=1,216$ unique outbound grids**
(10,944 constituent digit pairs). The $n=0$ state is after the initial quiet
interval, before any outbound batch. This clock counts all observations drawn
during the outbound drift, including its near-nominal start; it does not claim
each one contains equally novel information. Keep the frozen initialization
reference separate from this already-trained $n=0$ state.

Interpolate risk and cumulative learning time only between the recorded
post-batch states, onto integer $n=0,\ldots,N$, as in the original analysis.
This is interpolation across at most three unobserved grid counts, not
individual-image training. Do not overwrite acquisition points with later
return training. Finish all 928 batches, reporting return/final-quiet behavior
on the all-arrival clock of 0--3,712 grids. Shade regime boundaries on plots.
This acquisition clock applies only to changing replicas. Stationary paths
have no drift exposure and receive no novel-sample saving or $S$ estimate.

Use the same metrics with $N=1,216$ replacing 100:

- Rotated budget-averaged risk $A_j^{\rm new}=N^{-1}\sum_{n=1}^N R_j^{\rm new}(n)$,
  endpoint RMS, and one-pixel accuracy; old-test counterparts alongside them.
- Retention $H_j^{\rm avg}$ and $H_j^N$, measured relative to frozen initialization,
  with the existing root-after-averaging convention. Name the endpoint
  $H_{1216}$, never $H_{100}$. Final whole-path retention remains separate.
- Whole-path and regime-specific current-law RMS/one-pixel accuracy. These are
  tracking metrics, not curves to invert into sample savings.
- Common-reference sample equivalence against A at $N$: the existing persistent
  last crossing, $s_j=1-n_j/N$, matched-compute ratio $r_{T,j}$, and
  $S_j=s_j/r_{T,j}$. Include E versus B as the separately labeled secondary
  target. Keep weak/already-achieved-target checks, unsupported-crossing rates,
  and full/matched-budget retention qualifications unchanged.

Learning time includes the initial quiet interval and all method-specific
training/observer/transport work up to the matched outbound checkpoint.
Report initialization, development, evaluation, and actual study time
separately. All-arm equal-budget cost and matched-quality cost remain distinct.

**Simple explanation:** $S$ is the fraction of samples saved divided by the
compute multiplier for the same rotated-test quality. Here a sample is one
outbound grid, not a rare-component draw. This estimates conditional sample
equivalence along this schedule, not the effect of deleting those labels from
a different schedule. Do not pool old and new $S$ values or claim that their
difference isolates process breadth: both the process and acquisition budget
have changed.

**Reference-clock validity check (added during development, before confirmation):**
the readiness fixed-rotated risk falls and then rises near peak rotation. Show
each reference's own persistent crossing of its endpoint target alongside its
allocated $N$. Keep the prespecified $S$ calculation, but label it as saving
relative to that allocated endpoint; an identical method reaching the same
earlier crossing has no extra comparative efficiency. This reporting diagnostic
uses the existing scalar curves, changes no target/calibration/training, and
prevents treating a reference's late deterioration as a method-specific saving.

### Phase 5A: Versioned Implementation And Acceptance

Implement this protocol as a separately versioned configuration/entry point
under the existing package, with a new root such as
`cache/mnist_experiment/plan18/broad_drift_v1`. Preserve the original source
snapshot, configuration, artifacts, and report generations; its analysis must
remain reproducible. Share validated numerical helpers where compatible, but
never bypass an old source seal or overwrite a completed unit.

Add focused tests for the whole-stream law, unchanged quiet intervals and
ramp speed, independent source roles, deterministic pairing, outbound budget
and boundary indexing, and return observations not contaminating the outbound
curve. Cover B-A and the other paired contrasts, gain logging, separate noise
units, configuration collisions, and interrupted/resumed equivalence. Run a
tiny CPU pipeline and a bounded CUDA acceptance check before long trajectories.
No changes to `mathematical_overview.ipynb` are required.

### Phase 5B: Detectability, Numerical Readiness, And Timing

Run the two new development A/ridge-1 readiness paths; reuse them in calibration.
Repeat the bounded Phase 1 numerical checks at batches 32, 312, 464, and 616,
including solve adequacy, FD/transport consistency, fixed-anchor mapping, and
the FD gradient-strength scale. The wider process does not justify assuming
the old solve accuracy or gradient scale transfers. Do not repeat the optional
capability-fit experiment unless a concrete implementation defect requires it.

Add a **frozen-anchor, paired-source diagnostic** using the 128 diagnostic-role
grids per development replica. This is new derivative work, not an artifact-only
analysis, but needs neither new trained trajectories nor a dense Fisher.
For mapped observed-label score $z=D_{a,\epsilon}^{-1}s_a$, let $\mu_0$ be
its nominal expectation and $\mu_r(h)$ its rotated expectation. At the same
frozen initial model, compare the two stream means:

$$
\mu_{\rm rare}(t)=\mu_0+p_t[\mu_r(h_t)-\mu_0],
\qquad
\mu_{\rm broad}(t)=\mu_r(h_t),\qquad p_t=.125h_t.
$$

Their derivatives need not differ simply by a factor of $p_t$: the rare mean
also contains $p'_t[\mu_r(h_t)-\mu_0]$. Diagnose rather than assume attenuation.
Using the same fixed diagonal metric $D_{a,\epsilon}$, define
$V=\operatorname{tr}(D_{a,\epsilon}\operatorname{Cov}(z))$. Mixture variance is

$$
V_{\rm rare}(t)=(1-p_t)V_0+p_tV_r(h_t)
+p_t(1-p_t)\|\mu_r(h_t)-\mu_0\|_{D_{a,\epsilon}}^2,
\qquad V_{\rm broad}(t)=V_r(h_t).
$$

Estimate these with per-coordinate variances and weighted reductions, not
full covariance matrices. Pair source images and transform uniforms across
laws/states, analytically weight the rare mixture rather than adding Bernoulli
Monte Carlo noise, and retain finite-sample uncertainty. Evaluate nominal and
rotated scores at outbound indices $j\in\{64,128,192,256\}$ and $j\pm4$;
estimate per-batch mean-score motion from the paired difference divided by 8.
Report a paired bootstrap diagnostic and a split-half cross-product check on
motion energy: the squared norm of a noisy mean difference is biased upward.
Negative cross-product estimates indicate unresolved signal, not negative
physical energy. These uncertainty summaries concern this diagnostic sample,
not trained-method performance.

Compare motion with the **actual four-grid batch noise $V/4$**, separately from
the Monte Carlo error of the paired mean difference. A precise paired
diagnostic does not imply a live four-grid observation is equally precise.
The known schedule is available only to this offline assessment, never to the
online controller. This checks motion in the observer's score coordinates,
not the unknown population-optimum velocity or the validity of diagonal score
inversion. Keep the fixed-anchor discrepancy audit alongside it.

Instrument the unchanged live filter explicitly. Preserve $K_{t,i}$ above as
the scalar one-observation noise variance; call the two-component Kalman gain
$\mathbf G^{\rm KF}_{t,i}$ to avoid that notation collision:

$$
\mathbf G^{\rm KF}_{t,i}
=\frac{P^-_{t,i}H^T}{HP^-_{t,i}H^T+\mathcal R_{t,i}},
\qquad H=(1,0),\qquad \mathcal R_{t,i}=K_{t,i}/4.
$$

Log level/velocity gain means and coordinate quantiles, selected process-noise
multiplier $\eta_{\rm KF}$, innovation diagnostics, and actual predicted movement
versus batch noise. Plot **$\pi$** under that name. Distinguish the process-noise
multiplier, Kalman gain, and applied $\pi$; none is interchangeable with another.
Keep the actual inherited measurement-noise floor/order explicit in provenance.

Measure complete-path and operation costs, including increased evaluation
frequency. Use the existing bounded C/D timing probes only if needed. Weak
motion, imperfect mapping, or no apparent improvement is a scientific finding,
not an automatic stop or permission to alter the process. Proceed with the
best admissible calibrated recipes when numerically valid and affordable.

### Phase 5C: Retention Calibration And Freeze

Repeat Phase 2's bounded search on these two new, opposite-sign development
replicas. Re-estimate $\lambda_0$; retain ridge ratios $\{.1,1,10\}$, FD
multipliers $\{.1,1,10\}$, and shadow process scales
$\eta_{\rm KF}\in\{10^{-5},10^{-4},10^{-3}\}$. Preserve the staged shortlist,
maximum 34 complete development paths, and at most four optional refinement
paths. Every candidate still runs the complete 928-batch schedule.

Select using the new outbound $A^{\rm new}$, $H^{\rm avg}$ and $H_{1216}$.
Retain B's .50-pixel deterioration ceilings and the two-sided .10-pixel matching
band on both retention statistics. Inspect the retention/acquisition plot;
do not treat an insignificant old-error difference as equivalence or assume a
monotone ridge response. A best available unmatched recipe may proceed with
that qualification. The endpoint's replica scatter is not fixed by matching
development means; show it in confirmation.

Keep the common D/E process scale, innovation-score shortlist and declared
tie-breaks. Neither expected motion from the diagnostic nor larger observed
$\pi$ is a recipe-selection objective. Freeze all choices, metric definitions,
new seed namespaces, cohort size, and resource forecast before fresh outcomes.
Do not silently reuse earlier winners or tune on earlier confirmation seeds.

### Phase 5D: Fresh Paired Exploration

Target six new changing replicas plus two new stationary diagnostics, all five
arms: 40 complete paths. Retain the timing-only alternative of four changing
plus two stationary replicas, 30 paths, chosen before confirmation. Keep the
original sign balance, interleaved complete blocks, method-order randomization,
2,000 whole-paired-replica sign-stratified bootstrap draws, and descriptive-only
stationary reporting. Do not shorten the ramps or remove a control to fit time.

Prespecify **B-A**, E-B, E-A, C-B, D-B, and $E-C-D+B$ for rotated risk,
retention, and current tracking; show B-A first. Explain that the interaction
is on the reported outcome scale for retention-calibrated policies, not a
mechanistic proof at identical hyperparameters. Publish nominal 95% intervals,
paired-replica points, and sample/compute equivalence with its support checks.
No significance-based stopping or automatic addition of replicas.

The frozen-model reference, quiet periods, and stationary cohort distinguish
learning/tracking from deterioration without a changing stream. Diagnose
LFU magnitude/projection and $\pi$/motion/noise alongside predictive outcomes,
but do not promote estimator diagnostics to evidence of applied benefit.
Update the bounded eviction check to the first four initial-quiet observations
evicted after the outbound ramp begins and the first four outbound evictees,
at the existing 32/128-batch delays. This lets the old cohort's follow-up span
actual drift. These are in-sample retention diagnostics only, not extra
training or an independent test panel.

### Phase 5E: Readable Report And Closeout

Append a chronological Phase 5 section to `plan18_results.ipynb`, preserving
the earlier results and their qualifications. Open with the sparse-versus-broad
motivation, a short schedule diagram, and an explicit **1,216 outbound-grid**
efficiency budget. Near the notebook top, identify the active subphase, completed
and planned units, active hours, estimated remaining time, and any limitations.
Publish development and diagnostics as they complete, not only at closeout.

Keep the new section to four compact figure groups: process/detectability;
development retention versus acquisition; fresh prediction/sample equivalence;
and $\pi$/gain/motion diagnostics. Use small settings, paired-effects, and
efficiency tables, with units and lower/higher-is-better labels. Place per-replica
details and calibration candidate dumps in artifacts rather than long notebook
outputs. Separate old and rotated errors visibly; report savings fraction,
compute multiplier, and $S$ together, not as scattered prose.

Conclude whether broader drift produced incremental benefit over B and A,
whether retention supported that interpretation, and whether the observer
registered stronger motion. Increased motion without predictive improvement
would leave action/model mismatch open. A predictive improvement without a
clear diagnostic explanation remains useful, but would not prove the proposed
dilution mechanism. Neither outcome proves Kalman/$\pi$ optimality or generally
useful LFU composition. Preserve null results as evidence about scope.

### Compute, Recovery, And Authority

Phases 0-4 consumed about 7.38 active hours, leaving **about 12.62 hours of the
existing 20-hour Plan 18 ceiling**, before subsequent charged work. This is
not a new 20-hour allowance. Initial implementation is estimated at 2--4
engineering hours, separate from active compute; tests and numerical audits
are charged. Based on the completed study, provisionally budget:

| Subphase | Additional active wall-clock compute |
|---|---:|
| 5A: implementation acceptance | .3--.6 h |
| 5B: readiness, detectability, timing | .4--.8 h |
| 5C: bounded development calibration | 2--3.5 h |
| 5D: fresh paired exploration | 3--5 h |
| 5E: reporting and full closeout audit | 1 h reserved |
| Repair/restart contingency | 1 h reserved |
| **Total, including reserves** | **7.7--11.9 h** |

These are forecasts, not guarantees: stronger drift may increase solver cost,
and outbound evaluation is denser. Replace them with measured timing in 5B.
Use 25% contingency on remaining path costs and reserve at least one hour for
closeout before choosing six versus four changing replicas; do not add the
same contingency twice. Omit optional ridge refinement before reducing the
cohort. If the smaller complete allocation cannot fit, preserve development
findings and stop at the ceiling, reporting what additional allocation would
be needed rather than silently exceeding it.

Enforce the existing **500 GB total Plan 18 artifact/dependency ceiling** and
20 GB free-disk reserve across both roots. Budget new compact artifacts at
roughly 1--3 GB, replacing that forecast with measurements. Do not save dense
score covariance matrices or rendered streams. Maintain one cumulative active-
time ledger across both protocols, retries, reports, and reboots.

Require the existing `--resume`, graceful stop, single-writer lease, per-batch
transaction, pretraining checkpoint, source/configuration seals, atomic report
publication, and full final integrity checks. Recovery must restore the new
outbound count and diagnostic/selection state as well as model, archive, FIFO,
observer, seeds, and budgets. Partial paired blocks remain visible progress,
not complete inferential evidence. Record exact commands and current state in
`PLAN18_HANDOFF.md`; no manual reconstruction from this conversation.

After explicit execution authorization, Sol may implement, test, calibrate,
make documented numerical/software repairs, choose the timing-based allocation,
and close out this phase without routine user review. Material changes get new
provenance and a concise notebook note. The five methods, whole-stream law,
retention rules, and outcome definitions are not discretionary tuning knobs.
The user authorized execution on 2026-11-07.

### Completion Verification

All 77 Phase 5 trajectories completed: 37 development, six fresh five-method
blocks (30 paths), and two stationary five-method blocks (10 paths). The final
completion marker and eight-block pairing audit were written before the node
restart. A subsequent read-only audit verified all 107 completed units with
full checksums, unchanged scientific sources, and a valid sealed final report.

The Phase 5 reporter inherited a generic "pending" label for missing numbers.
At completion these numbers can instead be undefined because the target was
not reached or unsupported bootstrap crossings prevent a bound. Reporting now
uses "not estimable" and shows "none (execution complete)" for the active batch.
Seventeen focused reporting tests passed; the rebuilt statistical analysis is
byte-identical to the previous completed generation. All previous generations
remain intact. Total cumulative Plan 18 active compute after this check is
about 14.58 hours. No additional experiments were started.

### Phase 5 Conclusion: Retention Calibration Failed

**Phase 5 does not provide a valid five-condition comparison at comparable
retention.** The implementation completed its specified work, but the
calibration design failed the scientific purpose of that work.

The initial search matched the other arms to B while B itself violated its
absolute .50-pixel retention preference. Both permitted refinements then went
to B: ridge ratios 100 and 1000. The other arms were reselected from their
existing candidates, with no corresponding expansion of their ridge search.
The final ratios were A=10, B=1000, C=10, D=1, E=.1. Development retention
mismatches exceeded four pixels, yet the allowed unmatched fallback proceeded
to confirmation. This was an avoidable calibration/design failure, not a
restart failure or evidence that the numerical artifacts were corrupted.

The same two development replicas show the resulting trade-off. All B rows
use the same FD strength and fixed $\pi=.025$:

| Condition | Ridge ratio | Novel budget-averaged MSE, pixels squared | Old endpoint RMS deterioration, pixels |
|---|---:|---:|---:|
| A: Replay/EWC | 10 | 9.644 | 4.470 |
| B: STM | 1 | 9.055 | 4.517 |
| B: STM | 10 | 9.644 | 4.463 |
| B: STM | 100 | 13.504 | 3.513 |
| B: STM, selected | 1000 | 20.024 | .137 |

B's stronger ridge substantially traded new learning for old-data retention.
Comparisons involving B, including $E-C-D+B$, cannot isolate the intended
method contributions at comparable retention. They establish neither a
general STM failure nor an incremental benefit of LFU/adaptive $\pi$ over a
fairly calibrated STM incumbent. Other contrasts require their own retention
assessment; this qualification does not declare every recorded observation
invalid. The lower-ridge development evidence shows approximate A/B parity,
not a hidden established STM win.

Keep all trajectories, configurations, selection decisions, numerical results,
and earlier sealed report generations. Publish this conclusion in a new
report generation. Phase 6 addresses the failed comparison with a common
retention requirement and a mandatory calibration check before confirmation.

## Phase 6: Retention-Matched Recomparison

### Motivation And Question

The objective remains statistically and computationally efficient continual
learning. Phase 5 broadened continuous drift to make LFU and adaptive $\pi$
easier to assess, but its retention calibration made STM protect old data far
more strongly than the other methods. A precision-retention trade-off was
then presented in the place of the intended comparison at similar forgetting.
Additional confirmation replicas cannot repair that design defect.

**Question:** at approximately the same old-data deterioration as the existing
Replay/EWC control, which of the five methods learns new observations most
efficiently? Comparable forgetting is required for the primary comparison.
Low absolute forgetting remains desirable but is not imposed on only one arm.
This phase repairs calibration; it does not presume that any method will win.

Status: execution authorized by the user on 2026-11-08.
Astra specifies the design; Sol implements,
validates, calibrates, runs the fresh comparison, and records its limitations
without routine user intervention. All expensive work stays in resumable Python
entry points; the notebook only reads artifacts and renders summaries.

### Scope And References

Preserve Phase 5's whole-stream continuous rotation law, 928 batches of four
grids, 1,216-outbound-grid acquisition clock, likelihood, initialization,
model dimensions, FIFO32/replay2, eviction-only archive admission, diagonal
LTM EWC, FD sensitivity penalty, LFU, and predictable velocity-Kalman policy.
Keep all five arms A--E and their meanings. No change to the renderer, solver,
learning objective, observation count, or replay allocation is part of this
repair. Keep the frozen numerical policy, common FD strength
$\lambda_F=.5691316767656495$ for B--E, and common process multiplier
$\eta_{\rm KF}=10^{-5}$ for D/E. A--C retain $\pi=.025$.

The primary control is **A at ridge ratio 10**, the existing Phase 5 control.
Freeze this operating point before the new development trajectories. Its
development retention is the reference; B is no longer privileged by an
absolute ceiling. Tune ridge for B--E to attain comparable retention.
The old .50-pixel ceiling is descriptive only and does not enter selection.
Different ridge values are expected because their resulting regularization
also depends on each method's archive and $\pi$.

The existing two-replica development evidence contains this candidate set:

| Arm | Starting ridge | $\bar H^{\rm avg}$, pixels | $\bar H^{1216}$, pixels |
|---|---:|---:|---:|
| A: Replay/EWC | 10 | 2.362342 | 4.469951 |
| B: STM | 10 | 2.357051 | 4.462540 |
| C: STM + LFU | 10 | 2.333694 | 4.460042 |
| D: STM + adaptive $\pi$ | 1 | 2.367923 | 4.431826 |
| E: STM + LFU + adaptive $\pi$ | .1 | 2.414720 | 4.451717 |

The ranges are .081027 and .038125 pixels. These are development observations,
not fresh evidence or proof that matching will persist. They motivate a
targeted repair rather than a new factorial search. Absolute forgetting at
this operating point is substantial and must remain visible in the report.

Use a new implementation namespace, proposed `combined_memory.retention_match`,
and a new root, proposed `cache/mnist_experiment/plan18/retention_match_v1`.
Preserve historical scientific sources and completed runs. The Phase 5
development units may be referenced read-only by full identity/checksum;
never reclassify its confirmation or stationary trajectories as development
or new confirmation. Phase 5 results informed this repair, so the new study
is a historically informed follow-up rather than an independent replication
of its original design.

### Phase 6A: Implementation And Acceptance

Implement a separately versioned scheduler and reporter. Reuse the unchanged
learner, data law, observer, transport, and metrics, but **do not inherit the
old absolute-ceiling ranking or unmatched confirmation fallback**. Preserve
all source/configuration seals and distinguish imported from newly computed
units. Imported scalar summaries must be recomputed or checked against their
sealed trajectory exports with the unchanged 1,216-grid definitions.

Use the historical Phase 5 development replicas as two named development
members, then add two new opposite-sign development replicas. Use a new seed
namespace, proposed 18110806, with separate component seeds for these and all
new fresh/stationary replicas. Preserve the original study-wide disjoint
development/confirmation test-source partition. Fresh trajectories have new
initializations, streams, transforms, and sampling seeds; they do not imply
that every MNIST test image has never appeared in earlier reports.

For additional candidates on a historical development member, reproduce that
member's original initialization, assets, and component seeds, writing the new
candidate under the Phase 6 root. A new namespace must not silently change the
pairing within that member. All four members receive the complete schedule.

Add deterministic tests for:

- rejection of the Phase 5 selected tuple with B/ridge-1000;
- acceptance of the saved matched starting tuple at its recorded means;
- all-arm range checks, including two arms that are individually within .10
  of A but more than .10 apart from each other;
- equal candidate allowances, joint selection, and deterministic fallback;
- prevention of confirmation before a sealed passing calibration decision;
- missing/incomplete development members and imported-artifact corruption;
- resumed candidate selection without repeating or replacing completed work;
- historical/new seed pairing and exclusion of historical confirmation data;
- a tiny CPU/CUDA smoke pipeline with a deliberately failed-calibration path
  that publishes diagnostics and launches zero confirmation trajectories.

Benchmark the actual versioned pipeline on the first new development block.
Do not repeat the expensive score-detectability or model-readiness studies
unless acceptance exposes a numerical defect. Any necessary scientific repair
requires a new source identity and compatible development evidence before
freezing; it cannot be silently pooled with the old numerical policy.

### Phase 6B: Common-Retention Calibration

For each recipe, retain Phase 5's $H^{\rm avg}$ and $H^{1216}$: the former
is old-holdout RMS deterioration over outbound acquisition, and the latter
is deterioration at its endpoint. Take roots within each replica before
averaging. Use the same four development members for every selected recipe.
For a complete tuple $\mathbf c=(10,c_B,c_C,c_D,c_E)$, require

$$
\max_{j\in\{A,B,C,D,E\}}\bar H_j^\ell(c_j)
-\min_{j\in\{A,B,C,D,E\}}\bar H_j^\ell(c_j)
\le .10\ \text{pixels},\qquad
\ell\in\{\mathrm{avg},1216\}.
$$

This is a practical matching check on development means, not a significance
test or a guarantee of population equivalence. Require both inequalities;
neither average retention nor endpoint retention can substitute for the other.
Do not round means before the check, except for a documented floating-point
tolerance no larger than $10^{-8}$ pixels.

1. Verify and import compatible historical development candidates. Run the
   starting five-arm tuple on the two additional development members: ten new
   paths plus two shared initializations. Publish separate historical-pair,
   new-pair, and pooled-four retention summaries, per-replica points, and old
   risk curves. No member may be discarded because it spoils the match.
2. Form candidate tuples only from recipes complete on all four members.
   Among tuples satisfying both range checks, minimize equally weighted mean
   novel budget-averaged MSE across the five arms. Resolve ties within
   $10^{-8}$ by total measured learning time, then the lexicographic ridge
   tuple in A--E order, then stable recipe IDs. Fix A's recipe throughout the
   primary search. This does not optimize any preferred method's advantage.
3. If no tuple passes, give **each of B--E up to two additional ridge recipes**,
   rather than allocating two recipes globally. Previously explored but
   incomplete-on-four recipes consume the same candidate allowance when
   extended. Interleave refinement rounds across arms; do not exhaust one
   arm's allowance while another has an untested necessary bracket. Reserve
   the measured cost of the whole required round before launching it.
4. Choose proposals using observed retention responses: a geometric midpoint
   of neighboring positive ridges that bracket the target, or one adjacent
   decade when the explored boundary is insufficient. Permit zero as a lower
   boundary probe when .1 is overprotective; do not assume monotonicity. Keep
   proposals in $[0,1000]$. Record the bracket, response evidence, and reason.
   Run every admitted recipe on all four members, reusing exact historical
   paths where available. Failed numerical recipes remain recorded. Do not
   interpolate a performance estimate and treat it as an executed recipe.
5. If the primary reference still has no feasible tuple, the declared fallback
   control ridges are **1, then .1, then 100**, in that order. Stop at the first
   feasible reference. Evaluate A at each attempted reference on all four
   members; retain the same candidate pools and per-arm refinement limits.
   A fallback changes the shared retention operating point, not the .10-pixel
   matching tolerance. Choose it for feasibility, never for favorable method
   contrasts. Cost reservations and the cumulative ceiling still apply.
6. If no complete feasible tuple can be established within the bounded search
   and budget, finish with `calibration_infeasible` or `calibration_budget_limited`.
   Publish the frontier and required additional work. **Launch no fresh
   confirmation.** This is a comparison-validity check, not efficacy-based
   early stopping. Do not replace it with a "best unmatched" fallback.

Inspect the full old-error trajectories and record their maximum mean
separation over outbound acquisition. Similar summaries need not imply
pointwise retention matching. Show crossing or differently timed forgetting
and limit claims accordingly; do not silently add a new optimized metric.
Fresh sample-equivalence claims must still pass the retention check at the
actual comparison budgets.

Before any confirmation outcome, seal the selected tuple, all four-member
summaries, both range checks, reference/fallback identity, used and unused
candidate allowances, numerical/source identities, seeds, cohort allocation,
and resource forecast. A resumed scheduler must validate this seal and its
passing status before opening a fresh unit. If a later defect invalidates the
seal, stop the affected allocation and version the repair; never tune using
confirmation outcomes.

### Phase 6C: Fresh Paired Comparison

Default allocation: **four new changing replicas and two new stationary
diagnostic replicas**, all five arms: 30 complete trajectories. Balance global
signs within each cohort, pair initialization/Fisher/assets/seeds within each
replica, and rotate method execution order deterministically. The four-replica
changing allocation is a resource compromise and provides limited precision.
Do not add replicas automatically based on results or apparent significance.

Require the passing calibration seal and sufficient forecasted resource reserve
before the first fresh path. If even this allocation cannot fit, preserve the
calibration result and report the additional allowance needed. Do not silently
reduce the paired cohort, drop an arm, shorten the schedule, or exceed the cap.

Preserve the unregularized old, rotated, and current-law holdouts, frozen-model
reference, acquisition-risk definitions, and paired comparisons B-A first,
C-B, D-B, E-B, E-A, and $E-C-D+B$. Report nominal 95% intervals from the inherited
2,000 whole-paired-replica, sign-stratified bootstrap draws and show replica
points. These are exploratory intervals with a small number of independent
trajectories; minibatches are not independent replicates. The interaction is
between calibrated policies, not a mechanism estimate at identical ridge.

Reassess both retention differences on fresh replicas. A paired two-sided
95% interval wholly inside $[-.10,.10]$ pixels supports practical equivalence
for that contrast and statistic. An interval crossing a boundary leaves
equivalence unresolved; a nonsignificant difference alone is insufficient.
Publish all ten pairwise retention comparisons in a compact table, with the
prespecified scientific contrasts emphasized. Label intervals nominal rather
than claiming simultaneous all-arm population equivalence.

Preserve the existing sample-saving, compute-multiplier, and $S$ calculations,
support checks, and the reference's own first persistent crossing diagnostic.
Apply the inherited one-sided .10-pixel retention noninferiority checks at
full and matched budgets before describing savings as retention-supported.
That one-sided qualification allows better retention and is distinct from
the two-sided practical-equivalence assessment above. Never present an
unsupported inversion as zero savings. Stationary paths remain descriptive
and have no sample-saving statistic.

### Phase 6D: Human-Readable Report And Closeout

Append Phase 6 after the corrected Phase 5 conclusion in
`plan18_results.ipynb`. Open with the calibration failure and the question
being repaired. The near-top progress view must show the active subphase,
completed/expected paths, calibration pass/failure, cumulative and remaining
compute, and whether confirmation is authorized by the calibration seal.
Populate results as units complete; incomplete pairs are progress only.

Keep the main story compact: one settings/calibration table, development
retention versus acquisition with both retention metrics, fresh old/new risk
curves, a paired-effect table, and the sample/compute-efficiency plot and table.
Show the .10-pixel matching bounds and development versus fresh evidence
clearly. Put exhaustive candidate details in linked artifacts. Add compact
$\pi$/LFU diagnostics only where they help explain observed outcomes.

Conclude separately whether calibration passed, whether fresh retention was
comparable or uncertain, whether new-data performance improved, and whether
sample savings justified compute. Fairness is not defined by a favorable
STM result. Low absolute retention error is not established merely by matching
methods that all forget substantially.

Write a new immutable report generation and closeout marker after full artifact,
pairing, exposure-clock, and calibration-seal audits. Preserve Phase 0--5
history, including the Phase 5 failure conclusion and its unchanged numerical
results. Update `PLAN18_HANDOFF.md` with the actual terminal state.

### Compute, Recovery, And Authority

At drafting, Plan 18 has consumed about **14.58 active hours**, leaving about
**5.42 hours under its existing 20-hour cumulative ceiling**. This phase does
not reset or increase that allowance. The lean route relies on the already
matched development candidates; substantial refinement may require a separately
authorized extension before confirmation can run.

Measured Phase 5 fresh paths took about 221--358 seconds per trajectory, with
about 263 seconds per shared initialization. Budget B using its lower-ridge
development timing rather than assuming the fast ridge-1000 timing transfers.
Provisional additional active wall-clock estimates are:

| Subphase | Additional compute |
|---|---:|
| 6A: acceptance and smoke checks | .15--.30 h |
| 6B: two additional development blocks, no refinement | .9--1.1 h |
| 6C: four fresh plus two stationary blocks | 2.7--3.1 h |
| 6D: report and final audit | .20--.35 h |
| Restart/timing contingency | .8--1.0 h |
| **Lean total, including reserve** | **4.75--5.85 h** |
| Additional calibration, only if needed and affordable | roughly .3--3.5 h |

The upper lean forecast already exceeds the remaining allowance. Replace
forecasts with measured costs after the first new development block, reserve
25% on remaining trajectory/initialization costs plus .35 h for closeout,
and avoid double-counting the contingency row. Prefer completing a sound
calibration to spending the reserve on unmatched confirmation. If the fixed
confirmation allocation does not fit, report the shortfall without launching
it; obtaining more compute then requires user authorization. Engineering time
is separate, while tests, failures, reporting, and audits count as compute.

Retain the **500 GB family-wide artifact/dependency ceiling**, 20 GB free-disk
reserve, and cumulative active-time ledger across all Plan 18 roots. Reference
historical artifacts without duplicating checkpoints or charging old runtime
again. New work is provisionally 1--3 GB; measure it rather than relying on
this estimate. No dense Fisher or rendered-stream artifacts are needed.

Require `--resume`, per-batch atomic checkpoints, initialization recovery,
single-writer leases, graceful stops, immutable completed units, and atomic
report publication. Persist calibration candidates, pairing/member identities,
reference attempts, allowances, budget reservations, and the passing freeze
decision so a restart cannot repeat a search or bypass matching. Report-only
commands cannot train or repair scientific artifacts.

After explicit execution authorization, Sol may implement and execute this
design autonomously, including the specified bounded development refinements
and fallback references. Record material corrections in the notebook. Sol may
not widen the tolerance, impose a method-specific retention ceiling, substitute
an unmatched tuple, use confirmation for tuning, increase the allocation, or
raise the resource ceiling without a reviewed amendment.

### Phase 6 Execution Record

Execution was authorized on 2026-11-08. Phase 6A implemented a separate
`combined_memory.retention_match` scheduler, pure calibration module, verified
historical scalar references, and chronological reporter. Historical learning
sources remain unchanged. Development members 0/1 preserve their original
component seeds and initialization/assets; all new members use namespace
18110806. The old absolute retention ceiling does not enter selection.

Acceptance passed 142 focused unit tests, two cached-MNIST CPU lifecycle tests,
and a complete CUDA smoke study. Tests covered the saved matched/unmatched
tuples, both all-arm ranges, frozen policy settings, scalar corruption,
historical role separation, passing-seal enforcement, partially completed
refinement rounds and closed-reference replay, SIGTERM/resume, and deliberate
calibration failure with zero confirmation launches. Smoke studies use tiny
data/solve settings and a loose matching tolerance solely to test execution;
production retains the .10-pixel tolerance and full approved data schedule.

The CPU lifecycle study preserved completed units and the public notebook,
while its failed-calibration case published a terminal diagnostic notebook.
An early bounded smoke root predates the finalized scheduler source; it is
retained as a superseded acceptance artifact and is not a resume target.
All acceptance compute is charged to the shared Plan 18 ledger.

Production starts at `cache/mnist_experiment/plan18/retention_match_v1`, with
progress in `plan18_results.ipynb`. Calibration and the resource reservation
must pass before any fresh confirmation trajectory can begin.

During early development execution, a reporting-only review found that the
inherited stratified bootstrap can produce degenerate intervals with a
singleton fresh sign stratum. Before any fresh trajectories, the worker was
checkpointed and restarted to activate a presentation guard: means and
sample-equivalence points remain descriptive until both signs have two fresh
replicas. Intervals and supported-saving flags are withheld before then.
The final four-replica analysis, numerical paths, and calibration rules are
unchanged. Thirty-one focused reporting tests passed, including preservation
of the final inherited estimates and early descriptive points. Development
legends now also identify arms before all four members have completed.

All ten starting development paths completed. The pooled all-arm ranges were
.1050799153 pixels for $H^{\rm avg}$ and .0369907604 for $H^{1216}$, so the
unchanged .10-pixel gate correctly rejected the starting tuple. The automatic
proposer admitted E/ridge-.01 but omitted partially evaluated historical
responses when searching for a bracket. Its first admitted round is retained.

A development-only review used the paired historical member-0 E responses at
ridge .1 and 1. They bracket A's average-retention value, and both endpoint
responses already lie within .10 pixels of A. The geometric midpoint
$\sqrt{.1}=.31622776601683794$ was therefore sealed as E's second allowed
recipe, before any confirmation outcome. This is a proposal, not interpolation
of a result: the runner must execute all four members for each admitted round
and apply the original joint matching gate before selection. No completed
proposal or trajectory, numerical code, tolerance, or per-arm allowance changed.

The separate `retention_match.tools.bracket_review` entry point archives its
actual source and paired scalar evidence in the ordinary immutable
`decision-refinement-0-1` unit. This preserves the running scientific source
identity while recording the review explicitly. Six focused review tests and
32 reporting tests passed. A checkpointed restart activates its notebook note;
progress now counts all eight admitted refinement paths, including queued work.

The user explicitly approved a conditional increase of the cumulative compute
ceiling from 20 to **21.5 active hours** on 2026-11-08. Calibration remains under
the original allowance. If a passing four-member selection cannot fit the
unchanged four-changing/two-stationary confirmation allocation and reserve,
a separately versioned operational continuation may use the approved ceiling.
It must revalidate the passing seal, archived scalar evidence, scientific
identity, and unchanged replica allocation, and archive its source and the
approval in an immutable decision. No tolerance, learner, policy, outcome,
completed artifact, or scientific source may change. This is an additional
1.5 hours at most, not a reset of the cumulative clock.

Calibration passed on all four development members at 2026-11-08T19:36Z,
after both admitted refinement rounds finished. Selected ridges A/B/C/D/E
are 10/10/10/1/.31622776601683794. The all-arm spreads are .0754411206 pixels
for average deterioration and .0282754938 for endpoint deterioration, below
the unchanged .10-pixel limit. Absolute mean deterioration is still substantial
(about 2.3 pixels averaged over acquisition and 4.4 at its endpoint).

The original invocation closed without confirmation because its 13,813-second
forecast, including reserve, exceeded the remaining 20-hour allowance. The
approved continuation then revalidated the immutable selection and evidence,
archived its actual source and approval in `decision-resource-extension-21p5`,
and opened the unchanged confirmation allocation under the 21.5-hour ceiling.
Thirteen resource-continuation checks (including real-store immutable resume)
and 34 reporting checks passed. The prior budget closure remains recorded as
history; no scientific source, recipe, tolerance, or allocation changed.

### Phase 6 Outcome

The study completed normally at 2026-11-08T22:35:29Z: four fresh changing and
two stationary five-arm blocks, 30 confirmation paths. Including development,
there are 85 available trajectories (37 read-only imports and 48 newly run).
The final audit verified 71 completed units, all 46 frozen scientific source
hashes, paired exposure/assets, and the unchanged calibration seal. No GPU
experiment remains running. Total cumulative charge, including tests and
publication, is about **19.58 active hours**, below even the original ceiling;
the conditional extension was needed for the conservative admission forecast,
not because the full allowance was ultimately spent.

All ten fresh pairwise contrasts support practical equivalence of both
average and endpoint retention under the specified nominal 95% intervals.
This is not simultaneous population equivalence or low absolute forgetting:
average deterioration remains about 2.35 pixels and endpoint deterioration
about 4.4 pixels.

| Fresh contrast | Rotated budget-average MSE difference (px squared), nominal 95% CI | Interpretation |
|---|---:|---|
| B-A: STM versus Replay/EWC | .00346 [.00267, .00425] | Tiny worsening, no STM-alone acquisition benefit here |
| C-B: add LFU | .06157 [.04358, .07956] | Worse acquisition |
| D-B: add adaptive pi | -.04598 [-.05866, -.03330] | Better acquisition |
| E-B: add both | -.24976 [-.26251, -.23700] | Better acquisition |
| E-A: combined versus Replay/EWC | -.24630 [-.25939, -.23321] | About 2.66% lower acquisition risk |
| E-C-D+B: calibrated-policy interaction | -.26535 [-.28342, -.24729] | Favorable interaction, not an isolated mechanism estimate |

The preserved endpoint-budget inversion gives E-A **49.1% saved grids**
(nominal 95% interval **3.0--50.3%**), matched compute **1.33x**, equal-budget
compute **1.95x**, and **S=.370 [.016, .377]**. E-B's secondary target gives
49.0% [3.0, 50.2], matched compute .876x and S=.559 [.024, .579].
These are small-cohort exploratory estimates, not precise efficiency constants.

The nonmonotone trajectories are essential to interpretation: E reaches
*sustained* target quality at about 619 grids, whereas A's persistent crossing
is about 1,182. A first crosses near 629 but subsequently rises above the target.
The large inversion therefore describes earlier sustained quality, not twice
the initial learning speed. B's 1,181-grid crossing is essentially A's own
crossing; its roughly 2.9% allocated-budget saving does not demonstrate an
additional timing advantage. The original metric and all numerical results
remain unchanged; this qualification is explicit in the notebook conclusion.

Reporting generation `68c24ddf8e1f5ac8b76ac167` preserves byte-identical analysis
and figures from the computational closeout generation, adding only clearer
conclusion wording. Historical Phase 0--5 reports and the Phase 5 invalidity
conclusion remain intact. Stationary results are descriptive and have no S.
Final focused acceptance passed all 168 unit tests. The earlier actual
SIGTERM/resume CPU lifecycle checks and complete CUDA smoke study also passed.

### Post-Run Artifact Review And Diagnostics

At the user's request, reporting generation `8960658bfe81951b6668b306` adds
Phase 6's **Pi, Gain And Motion Diagnostics**, using the existing Phase 5
renderer on Phase 6's completed scalar trajectories. It also clarifies the
sample-saving estimator and percentile interval. No training, calibration,
or estimand changed. Analysis and the two earlier Phase 6 figures remain
byte-identical; the historical notebook cells and both report seals passed
verification. All 36 focused reporting tests passed, including fresh-only
diagnostic aggregation and distinct one-grid/batch noise units.

The 49.0637% E-A estimate is one minus the persistent crossing of the mean E
trajectory against the mean A endpoint target divided by the allocated
1,216-grid budget; it is not the median of replica savings. Reconstructing the original 2,000
paired, sign-stratified bootstrap draws gives the existing interval
[2.9804%, 50.2737%]. In 757 draws (37.85%), the persistent crossing moves to
the late branch, giving about 3% savings; the other draws give about 48--50%.
The bootstrap median is 48.3136%, distinct from the reported point estimate.
There are only nine distinct replica-weight compositions with two replicas
per sign, so this interval has limited small-cohort resolution. Calibration
is held fixed rather than rerun within the bootstrap.

The ridge hypothesis has supporting development evidence. All rows below
average the same two paired development members within their respective study;
the FD strength is fixed within each study's displayed ridge comparison.

| Study | Ridge c | Replay/EWC acquisition MSE | STM FD acquisition MSE |
|---|---:|---:|---:|
| Plan 15 Phase 12, head r1 | 1 | 2.94036 | 2.65984 |
| Plan 15 Phase 12, head r1 | 3.16228 | 2.82322 | 2.83062 |
| Plan 18 broad-drift development | 1 | 9.10388 | 9.05503 |
| Plan 18 broad-drift development | 10 | 9.64414 | 9.64376 |

Sources are Plan 15's `stm_capacity_records_v2/report_snapshot.json` and the
sealed historical development imports in `retention_match_v1`. At c=1 in
Plan 18, B-A average/endpoint retention differences are -.01483/-.00418 pixels:
the small acquisition benefit is not accompanied by worse mean retention.
These are development leads, not fresh confirmation. Plan 15's successful
head-r1 recipe used c=1 and FD strength 3.34056; Phase 6 uses c=10 and .56913.
Raw coefficients alone do not compare effective penalties because archive
scale and objective gradients differ. The rare-drip versus whole-stream
process and rank-64-plus-diagonal versus diagonal EWC also differ. Lower ridge
is therefore a credible follow-up, not an established explanation or a promise
to recover the earlier 41.4% saving. No additional trajectories were launched.

## Phase 7: Low-Ridge STM With Recalibrated Retention

### Motivation And Question

The objective remains sample-efficient continual learning under limited replay
and compute, not recovery of a preferred ranking. Phase 6 repaired the unequal
retention of Phase 5 and found a favorable combined policy, but the STM-only
arm B contributed essentially no acquisition benefit at ridge ratio 10.
This contrasts with Plan 15's successful head-rank-one STM at ridge ratio 1.
The historical development evidence is more specific than that cross-study
comparison: raising ridge suppressed the incremental FD benefit within both
studies, as tabulated in the post-run review above.

Simply replacing Phase 6's B with ridge 1 would not answer the fair-comparison
question. On the same historical development members 0/1, B at ridge 1 has
average old-data RMS deterioration 2.491026 pixels, versus 2.362342 for the
frozen A at ridge 10. Their .128684-pixel gap exceeds the .10-pixel matching
tolerance, although the endpoint gap is only .047056 pixels. Substituting
this B into the existing five-arm tuple gives an average-retention range of
.157332 pixels. Comparing B and A *both at ridge 1* instead gives gaps
-.014830 and -.004179 pixels; that is a promising starting point, not proof
that all five methods will match on four development members or fresh paths.

**Question:** when B is allowed to operate at ridge ratio 1, does its STM
penalty improve acquisition and sample efficiency over Replay/EWC at comparable
old-data retention, and do LFU or adaptive pi add value at that same operating
point? The user prefers spending compute on recalibrating the comparison to
preserving the cheaper but more restrictive Phase 6 reference.

Phase 6 is not invalidated. This is a historically informed experiment at a
different retention operating point. A favorable result would identify useful
conditions for the technology, not prove ridge 1 universally optimal or show
that the difference between phase summaries is a causal ridge effect.

**Status:** design for review; no execution is authorized by this draft alone.
Astra supplies this design; after approval, Sol implements and executes the
whole phase, publishes rolling results, and records material corrections.
No change to `mathematical_overview.ipynb` is part of this work.

### Why Ridge One Is A Meaningful Probe

These are local calculations in the existing parameter chart, not a new
Fisher estimand or an instruction to alter the learner. For a nonzero archived
LTM diagonal $D_t$, let $s_t=\operatorname{tr}(D_t)/p_L>0$. Its least-squares
isotropic approximation is

$$
\underset{a\ge0}{\operatorname{argmin}}\ \|D_t-aI_{p_L}\|_F^2=s_t.
$$

Thus $c=1$ sets $\kappa_t=s_t$: the added isotropic curvature equals the
average archived Fisher curvature. It is a dimensionless, trace-scaled
reference, not an arbitrary absolute penalty. The LTM regularization matrix
can be written

$$
D_t+c s_t I_{p_L}
=(1+c)\left[(1-\gamma_c)D_t+\gamma_c s_t I_{p_L}\right],
\qquad \gamma_c=\frac{c}{1+c}.
$$

At $c=1$, the trace-normalized shape is halfway between the diagonal Fisher
and its isotropic approximation; at $c=10$, the isotropic share is $10/11$.
Weak directions receive a floor without requiring a reliable smallest
eigenvalue. But the implemented penalty is **not trace-normalized**: its LTM
trace is multiplied by $1+c$. Ridge 1 therefore doubles the archive penalty
trace and ridge 10 multiplies it by eleven, at fixed $D_t$ and $\pi_t$.
The same $\kappa_t$ also constrains STM movement about its pre-arrival anchor.
Ridge can both stabilize weak directions and suppress the freedom through
which STM helps. This offers a mechanism hypothesis for the observed response,
not evidence that all of the learning objective is well-conditioned.
Ridge is not generally redundant with pi: at fixed archive, pi rescales both
the Fisher-shaped and isotropic terms, whereas c changes their relative
strength. Pi also changes the subsequent archive blend. Exceptional isotropic
or scalar cases do not make these controls interchangeable in this model.

The shrinkage connection is real, but does not supply a universal coefficient.
[Ledoit and Wolf (2004)](https://www.ledoit.net/Well-conditioned2004.pdf) derive
an estimated, asymptotically optimal covariance shrinkage intensity under a
quadratic matrix loss; their result does not select $c=1$ for our moving EWC
archive or optimize retained/new predictive error. Likewise, Fisher damping
has an optimization motivation, as discussed by
[Martens (2020)](https://arxiv.org/abs/1412.1193), but our ridge remains in the
statistical objective and changes its solution. It is not merely an optimizer
damping term that can disappear upon convergence.

A simple derivation illustrates why a universal optimal $c$ is unlikely.
Consider one local coordinate with fixed old anchor $a$, unbiased fresh estimate
$y=\theta^\star+\varepsilon$, $\mathbb E\varepsilon=0$,
$\operatorname{Var}(\varepsilon)=v$, new-loss curvature $h>0$, and old diagonal
curvature $d\ge0$. Hold $h,d,s,\omega$ fixed, with $s,\omega>0$, and minimize

$$
\frac h2(\theta-y)^2+
\frac{\omega(d+cs)}2(\theta-a)^2.
$$

Writing $\delta=\theta^\star-a$ and
$w=h/[h+\omega(d+cs)]$, the parameter MSE is

$$
\mathbb E[(\widehat\theta-\theta^\star)^2]
=(1-w)^2\delta^2+w^2v.
$$

For $\delta^2>0$, its unconstrained optimal weight is
$w^\star=\delta^2/(\delta^2+v)$. With $c\ge0$, the corresponding optimum is

$$
c^\star=
\max\left\{0,\frac{h v/\delta^2-\omega d}{\omega s}\right\}.
$$

The zero-signal, positive-noise limit favors freezing in this toy problem;
other signal/noise ratios favor smaller penalties. Even here, $c=1$ requires
a particular balance, not an identity. The real model adds coupled coordinates,
an uncertain anchor, replay, FD regularization, and changing curvature. This
argument motivates future estimation of regularization from uncertainty but
does not authorize plugging the merged-coordinate Kalman moments into the
raw LTM/STM chart without deriving their relationship. One scalar also cannot
generally match all coordinate-specific optima. Coordinate rescalings change
the meaning of an isotropic ridge, so this is not a chart-invariant optimum.

**Scope decision:** test the interpretable fixed value now. Do not add a new
ridge controller, shrinkage estimator, covariance reference study, separate
STM/LTM ridge coefficients, or further tuning dimension to this phase.

### Conditions And Fixed Scientific Contract

Preserve the Phase 6 whole-stream rotation law, padded rendering, 928 batches
of four grids, 1,216-grid outbound clock, full return/quiet trajectory,
54-output Gaussian likelihood, initialization, 3,268 LTM plus 108 STM parameters,
diagonal EWC, FIFO32/replay2, and equal incoming/replay mean-loss weights.
Archive only actual evictees; transport only the old LTM diagonal along the
accepted displacement. Keep the existing frozen numerical policy and all
held-out evaluation roles. A retains the same trainable STM coordinates as B.

| Arm | Method | Ridge ratio | FD strength | Adaptation weight |
|---|---|---|---|---|
| A | Replay/EWC control | Development-calibrated | 0 | .025 |
| B | STM FD reference | **Fixed at 1** | .5691316767656495 | .025 |
| C | STM FD + LFU | Development-calibrated | Same as B | .025 |
| D | STM FD + adaptive pi | Development-calibrated | Same as B | Velocity Kalman |
| E | Combined | Development-calibrated | Same as B | Velocity Kalman |

Keep the common D/E process multiplier $10^{-5}$. No FD-strength or Kalman
search is included: this tests the low-ridge STM recipe and comparably retained
alternatives, not independently exhaustive hyperparameter optimization.
There is no fallback to B at ridge 10 or 3.16228. The calibrated A remains
the **sample-equivalence reference** even though B now defines the development
retention target. These two reference roles must be distinct in code and prose.

Record and plot $c$, $s_t$, $\kappa_t$, $\omega_t$, and $\omega_t\kappa_t$.
Raw $c$ alone does not measure regularization: Phase 6 D's outbound mean
$c\omega_t$ was about 264 versus B's 390, whereas B at ridge 1 and fixed
$\pi=.025$ has $c\omega_t=39$. The archived scale can also differ by method.

### Phase 7A: Versioned Implementation And Acceptance

Use a separate namespace such as `combined_memory.low_ridge` and a new root
`cache/mnist_experiment/plan18/low_ridge_v1`. Reuse the frozen learner and
validated artifact/reporting utilities without changing historical scientific
sources, seals, or completed units. Do not fork the numerical learner merely
to change calibration or seeds.

Retain the four Phase 6 development members, with their exact initialization,
assets, global signs, and named component seeds. For members 0/1, resolve the
original Phase 5 dependencies; for 2/3, resolve the Phase 6 dependencies.
All new candidates must reproduce those streams, not obtain different seeds
because a new protocol name was introduced. Deduplicate imported trajectories
by original root/unit identity and checksum, including transitive imports.
An old candidate with matching settings is reusable only if numerical policy,
initialization, stream, and all pairing hashes also match.

Only historical **development** trajectories enter calibration. Keep prior
confirmation outcomes as context, never convert them into development or call
them fresh Phase 7 evidence. Use a new namespace, proposed seed 18110907, for
all Phase 7 confirmation initializations and named random components. Preserve
the established development/confirmation source partition; fresh means new
replica draws, not a claim of never-before-used MNIST source images.

Acceptance must cover: mandatory B/ridge-1 reference; rejection of the unchanged
Phase 6 tuple as a ridge-1 substitute; both all-arm range checks; partial
historical brackets; immutable import deduplication; exact pairing for all
four development members; candidate-round resumption; budget reservations;
and a failed matching decision that launches zero confirmation paths. Test
new seed separation, post-batch/outbound indexing, and retention-reference B
versus efficiency-reference A. Run a tiny CPU SIGTERM/resume pipeline and one
bounded CUDA smoke check. Keep these acceptance inputs and their tolerance
overrides separate from production. Report-only execution must not train.

### Phase 7B: Complete The Fixed STM Reference

Import B/ridge-1 development members 0/1 at the fixed FD strength and run the
two missing members 2/3. Reuse initializations rather than fitting them again.
Seal the four-member B reference before selecting any other arm. Its measured
$\bar H_B^{\rm avg}$ and $\bar H_B^{1216}$ define the new retention target;
do not choose targets from whichever fresh or development metric looks best.

Publish the four individual old/new paths, the pooled reference, its contrast
with historical B/ridge-10 on these same members, and the measured runtime.
The B1-B10 contrast is a paired **development ridge-response diagnostic**, not
fresh efficacy evidence. Inspect existing solve and FD diagnostics for a
numerical defect, but do not reject B because its predictive result disappoints.
Valid finite results proceed to calibration whether or not they favor STM.
Only an actual numerical defect requires a versioned repair and revalidation.

### Phase 7C: Recalibrate A, C, D, And E

Evaluate all selected recipes on the same four development members. With
$c_B=1$ fixed, retain the Phase 6 requirement

$$
\max_{j\in\{A,B,C,D,E\}}\bar H_j^\ell(c_j)
-\min_{j\in\{A,B,C,D,E\}}\bar H_j^\ell(c_j)
\le .10\ \text{pixels},\qquad \ell\in\{\mathrm{avg},1216\}.
$$

Use the unchanged metric definitions and root within each replica before
aggregating across replicas. The code's historical `H100` alias must be shown
as $H_{1216}$ in this report. An endpoint match cannot substitute for an
average match, and individual proximity to B cannot substitute for the full
five-arm range. Do not reinstate the old absolute .50-pixel ceiling.

1. Import all compatible candidate evidence. Start by completing A/ridge-1,
   C/ridge-1, D/ridge-.1, and E/ridge-.1 on all four members. These are search
   starts, not asserted matches. Already complete alternative recipes remain
   eligible, except that B is always the fixed ridge-1 recipe.
2. Search the complete candidate tuples. Among those passing both range
   checks, minimize the equally weighted mean acquisition MSE across all
   five arms, as in Phase 6. Break ties within $10^{-8}$ by total learning
   time and then the lexicographic ridge tuple. Never maximize B-A or E-A
   gains to choose a control, and never select an incomplete candidate.
3. If none passes, admit up to **five additional recipe proposals per tunable
   arm**, beyond its starting recipe. Completing a previously partial recipe
   counts as a proposal; fully imported four-member recipes cost no new slot.
   Interleave the arms rather than exhausting one arm first. Each admitted
   recipe must finish all four members before selection or replacement.
   Persist proposals and remaining allowances so a restart cannot repeat them.
4. Use measured retention responses, including partial historical evidence
   on the same available members, to propose positive-ridge geometric
   midpoints or adjacent boundary decades in $[10^{-4},1000]$. Zero is one
   admissible boundary probe when the lowest positive recipe still retains
   too much. Partial evidence may propose a recipe, but never certify it.
   Do not require both retention statistics to bracket B exactly: one may
   already lie in its allowed band while the other needs adjustment.
5. For nonmonotone or competing responses, rank untried interior midpoints
   by the smallest predicted worst two-metric mismatch to B using
   piecewise-linear interpolation in log ridge. Use the same matched member
   subset on both sides and against B. Prefer four-member evidence, then
   more covered members, then stable recipe IDs on score ties. When no
   interior proposal predicts improvement over its better endpoint, probe
   an adjacent boundary decade, starting from the observed edge closest to
   B in worst two-metric mismatch. Do not extrapolate a boundary prediction
   or interpolate through zero on a log scale. Inspect the retention/acquisition
   plot, record any justified alternative proposal, and run it before making
   any inference. No surrogate response is an observed result. All proposals
   share the same allowance.
6. Recheck the entire candidate pool after each completed round. Freeze on
   the first passing round; there is no obligation to spend the whole search
   allowance. Do not stop or expand confirmation based on a promising
   acquisition effect. If the bounded search cannot match both statistics,
   publish the frontier and close as `calibration_infeasible` or
   `calibration_budget_limited`. Do not raise B's ridge, relax the tolerance,
   cherry-pick members, or run an unmatched five-arm confirmation.

All four tunable arms receive the same maximum opportunity, although only
arms whose responses need exploration must spend it. A single ridge may not
match two time-dependent retention summaries; an infeasible result is a
design limitation, not evidence against STM at ridge 1. More search or an
additional tuning dimension would need a reviewed amendment.

Show full old-data error paths and the maximum mean pointwise separation
during acquisition. Summary matching does not guarantee pointwise matching.
Before confirmation, seal B's fixed recipe, the four-member reference, every
selected recipe, both range checks, all search decisions, numerical/source
identities, new seeds, fixed cohort allocation, and measured resource reserve.

### Phase 7D: Fresh Paired Comparison

Run **eight new changing replicas and two new stationary diagnostic replicas**,
all five arms: **50 trajectories**, with four changing replicas per global
rotation sign and one stationary replica per sign. This doubles the Phase 6
changing cohort to improve the poorly resolved sample-saving distribution;
it does not promise a particular confidence width or significance.

Pair initialization, initial Fisher, streams, holdouts, and independently
named component seeds within each replica. Rotate execution order
deterministically. Start only after the passing calibration seal and a resource
reservation for the full allocation plus closeout. Always finish complete
paired blocks; do not add replicas for significance or terminate for a small
effect. Prior Phase 6 confirmation paths do not enter these averages or CIs.

The first question is **B-A** at comparable retention. Retain C-B, D-B, E-B,
E-A, and $E-C-D+B$ as additional contrasts. These compare calibrated policies;
different ridge choices prevent interpreting the interaction as an isolated
mechanistic LFU-by-adaptation effect at common regularization.

Preserve unregularized holdout MSE/RMS, one-pixel accuracy, acquisition-risk
averages, return/quiet performance, and frozen-initialization references.
Use the existing 2,000 whole-paired-replica, sign-stratified bootstrap draws
for nominal 95% intervals, withhold interim intervals while either sign has
fewer than two completed replicas, and show individual replica points.
Keep all ten pairwise retention contrasts; practical equivalence requires
the two-sided interval inside [-.10, .10] pixels for each statistic. Do not
equate nonsignificance with equivalence or claim familywise coverage.

Keep sample equivalence against the **newly calibrated A** at $N=1,216$;
E versus B is the separately labelled secondary target. Recompute the mean
reference endpoint target and the method's sustained last downward crossing
inside every bootstrap draw. Preserve the full/matched-budget retention
noninferiority checks, weak-target checks, unsupported-crossing counts,
measured learning-compute multipliers, and $S$ definition. Unsupported inverses
are not zero savings and failed bootstrap draws must not be discarded.

Show first versus persistent crossings for the reference and treatments.
Include a compact bootstrap histogram of saved fractions for B-A and E-A
when estimable, alongside the interval, so early/late crossing mixtures
cannot hide behind a single impressive point. Clearly distinguish savings
from mean curves from the median of bootstrap savings. Stationary diagnostics
have neither an acquisition clock nor an $S$ score. Phase 6/7 savings use
different calibrated A targets and are not directly comparable effect sizes.

### Phase 7E: Readable Report And Closeout

Append Phase 7 chronologically to `plan18_results.ipynb`, leaving Phase 6's
numerical results and qualifications intact. Introduce the ridge-suppression
hypothesis and why recalibrating all conditions was necessary. The near-top
progress display must distinguish imported/new development paths, active
proposal round, completed/expected paired blocks, calibration state, and
additional versus family-cumulative time. Refresh after completed units and
at least every 15 minutes during long work without replaying all checkpoints.

Keep the main section compact, following Plan 13's readable style:

- One settings table, including B's fixed ridge, all other selected ridges,
  effective regularization summaries, and the two reference roles.
- Development retention versus acquisition, with the two retention measures,
  .10-pixel tolerance, four-member summaries, and missing coverage explicit.
- Fresh old/new/current prediction trajectories, one concise paired-effect
  table, and a compact retention-equivalence table.
- Sample/compute-efficiency plot and table, with the crossing diagnostic and
  small bootstrap histogram where needed to interpret savings.
- The requested **Pi, Gain And Motion Diagnostics**, plus compact ridge-scale
  and STM/LTM movement summaries from existing scalar diagnostics.
- A conclusion separating calibration, fresh retention, acquisition, sample
  savings, and compute cost. No recovered STM benefit is assumed.

Explain $S$ simply: **fraction of samples saved divided by the compute
multiplier at the same new-data test quality**. Keep cohort labels, the
outbound-grid clock, held-out evaluation status, and lower/higher-is-better
directions explicit. Put exhaustive candidate tables and numerical logs in
linked artifacts, not a notebook wall of numbers. Record material corrections
without deleting their earlier results. Publish a new immutable report
generation and closeout marker after checksum/source/pairing/calibration
audits, then update `PLAN18_HANDOFF.md`.

### Compute, Recovery, And Authority

This is additional work, not a request to fit a new experiment inside the
roughly two hours remaining under the historical 21.5-hour ceiling. The
**proposed Phase 7 allowance is up to 20 additional active compute hours**,
available upon approval to execute this phase. Record the live family charge
at launch and its additional allowance in an immutable authorization entry;
retain the cumulative ledger rather than resetting prior usage. At the
current approximately 19.6-hour baseline, that means a family ceiling of
approximately 39.6 hours, with the exact number fixed at launch.

Phase 6 fresh trajectories took about 224/289/337/284/349 seconds for A/B/C/D/E,
including trajectory execution and artifact work, with about 267 seconds
per shared initialization. Lower-ridge solves can cost more. These timings
motivate, but do not guarantee, the following **additional active wall-clock**
forecast; software authoring time is separate:

| Subphase | Forecast |
|---|---:|
| 7A: import checks, tests, CPU/CUDA smoke | .25--.50 h |
| 7B: two missing B/ridge-1 paths and reference review | .15--.25 h |
| 7C: starting recipes and targeted recalibration | 1.5--4.5 h |
| 7D: eight changing plus two stationary five-arm blocks | 5.0--6.5 h |
| 7E: final report and integrity audit | .25--.50 h |
| Timing/restart contingency | 1.5--2.5 h |
| **Expected total including contingency** | **about 9--15 h** |

If calibration is difficult, use the additional 20-hour allowance rather
than prematurely discarding the idea, but reserve the entire fixed confirmation
allocation. After the first new paths, replace estimates with measured
per-arm costs. Reserve 25% on remaining initialization/trajectory costs plus
.5 h for closeout; this replaces, not duplicates, the contingency row. Charge
tests, failed attempts, reports, and audits. Reused historical compute is not
charged twice. If the full fixed confirmation cannot fit, preserve calibration
and report the shortfall before opening fresh units; do not silently shrink
the cohort or exceed the allowance.

Retain the **500 GB family-wide artifact ceiling and 20 GB free-disk reserve**.
Reference immutable historical data instead of copying it, keep scalar exports
notebook-friendly, and avoid dense matrices or rendered-stream dumps. New
artifacts are provisionally 1--4 GB, to be measured during acceptance. Require
`--resume`, atomic per-batch/initialization checkpoints, single-writer leases,
graceful stops, and immutable completed units. Persist the fixed B reference,
candidate coverage/proposals, completed rounds, budget reservations, seals,
and report pointer so an unexpected restart neither repeats calibration nor
changes its scientific meaning.

After execution approval, Sol may complete these subphases autonomously,
including recorded numerical/engineering corrections and the bounded
development search. Never alter completed artifacts or silently change the
fixed B recipe, data law, tolerance, five methods, confirmation allocation,
or efficiency estimand. If a repair changes scientific outputs, version it
and rerun affected development before freezing; do not mix incompatible
trajectories. This draft neither starts GPU work nor modifies Phase 6.
