# Implementation Plan 19: Sample-Efficient Continual PiCar Learning

**Motivation:** find practical ways to keep learning when new experience,
replay memory, GPU memory, and update time are scarce. The MNIST studies,
especially Plan 15, motivate an STM/LTM architecture that can save new samples
without requiring indefinite replay. Robotics is the next test of that idea:
does it help a language-conditioned controller adapt to continuous changes
while retaining a useful earlier capability? Statistical efficiency and
computational efficiency are both objectives. A smaller training loss, a
better Fisher estimate, or a more attractive theoretical interpretation is
not sufficient evidence of practical success.

**Status:** Phase 0 bounded implementation acceptance passed on 2026-11-09:
280 tests, paired CPU smoke/recovery, and actual quantized-Qwen CUDA checks.
Phase 1 diagnostics and two Fix 1 passes are recorded. Native/split parity,
revised instruction contrasts, and short cooperative timing pilots pass;
navigation and reward readiness remain unresolved. **Phase 1 Fix 2** is now
implemented and its bounded diagnostics are complete. Generated rewards are
valid but coarse; ported Qwen reached the ball in 3/6 unobstructed trials and
the obstacle task achieved 0/12 across two interfaces. Both real-time student
pilots had zero non-startup fallbacks. These are readiness checks, not learning
results. The versioned task/reward revision leaves earlier artifacts intact.
Phases 2--5 and physical robot motion have not started. Revised conditional
whole-study forecasts are 21.4--30.8 h nominal, or 11.6--17.2 h at the smaller
permitted size. Task readiness is not yet established; neither forecast is
authorization to proceed. See the Phase 1 execution
record and `PLAN19_HANDOFF.md` for findings and recovery notes. The theory below is
provisional and belongs here until
experiments establish its value. Do not modify
[`mathematical_overview.ipynb`](../../mathematical_overview.ipynb).

**Next amendment:** Phase 1 Fix 3 (2026-11-11) is drafted below for review:
modest centering rewards, a gradual barrier-placement curriculum and a bounded
student trainability pilot. It has not been implemented or executed.

**Report:** `numerical_experiment/picar/plan19_results.ipynb`, created during
implementation. Follow the concise, visual reporting style of Plan 13:
progress near the top, phases in chronological order, and a conclusion at the
bottom. The notebook reads artifacts only.

**Operating model:** Phase 0 uses Astra 6 Extra High for the numerical slice
and Sol 6.1 Extra High for the surrounding implementation, integration, and
operational tests. Prefer Sol as the coordinator, with bounded Astra
assignments rather than Astra supervising every engineering step. Sol then
profiles, calibrates, runs, repairs, and closes out the experiment. Explicit
sub-agent model selection avoids requiring the user to switch models between
assignments. Record material mathematical or experimental changes explicitly.

**Resources:** aim for 20 cumulative active wall-clock compute hours, but
reforecast after profiling rather than promising an unmeasured VLM runtime.
The current planning range is 15--33 hours, excluding implementation labor.
An extension requires approval; the user's willingness to revisit the budget
is not an unlimited allocation. Use an external archive root supplied as an
argument, with a bounded local working set. See the resource table below.

## Research Question And Boundaries

At comparable retention and with the same model capacity and replay resources,
does age-dependent STM finite-difference KL regularization reduce the new
robot experience needed to acquire a changed task? What additional compute,
latency, and memory does that saving require?

Start with one task, one smooth outward-and-return drift schedule, and two
training conditions. Do not combine this first robotics study with LFUs,
adaptive $\pi$, multiple curvature estimators, or a large architecture search.
They remain possible later extensions, not prerequisites for useful results.

| Condition | Replay and diagonal LTM EWC | Trainable LTM and STM banks | STM FD-KL |
|---|---|---|---|
| Replay/EWC control | Yes | Yes | $\lambda_{\mathrm{FD}}=0$ |
| STM FD | Yes | Identical capacity | Development-calibrated positive strength |

The control has the same extra parameters as STM FD. It is not a smaller
LTM-only network. A frozen initial student is an evaluation reference; Qwen
and a scripted geometric controller provide readiness diagnostics, not extra
full-budget training arms. Report neither as an equal-cost learning control.

Internal precedents are [Plan 15](../../mnist_experiment/plan15.md) for the
STM comparison and sample-efficiency analysis,
[Plan 18](../../mnist_experiment/plan18.md) for memory-block interpretation,
and [Plan 13](../../mnist_experiment/plan13.md) for readable reporting.
Their effect sizes are motivation, not forecasts for this changed task.

Development-only retention calibration is mandatory. Preserve a failed
calibration as an informative trade-off, but do not turn it into a claim of
sample savings at equal retention. A useful negative or qualified result is
preferable to an apparently strong result with an uncontrolled old-task gap.

### Provenance And Deliberate Changes

This plan uses the supplied *PiCar continual-learning handoff* as design input,
not as executable instructions. Its ZIP SHA-256 is
`980eb641d10a0187690609ee96c6674b18f24c2f4def5ccbb0b141a650fbe46a`.
The inspected `kl-projection` checkout was at commit
`b0bdccae09ebb60ddabbf30fa63c87da20007150`.

The discussion supersedes several original handoff choices:

- Use continuous texture and steering-response changes, not an abrupt A/B switch.
- Preserve the eight-action to four-control weighted mapping, including camera controls.
- Let Qwen collect the initial demonstrations; distill once, then let the LSTM controller learn online.
- Give the LSTM an actually language-conditioned transformer feature, not only a visual/action prefix.
- Restrict transformer adaptation to top-$L$ blocks; profile actual activation memory.
- Replace the legacy composite actor/critic objective with the explicit likelihood below.
- Maintain simulator and physical backends under one interface, including robot-side safety software.
- Run physics independently of inference and optimization; account for delayed commands.

This is a new robotics specification. It does not revive the retired synthetic
Gaussian auxiliary-score experiment. The conditional Gaussian below models
action coordinates in a specified likelihood; it does not assume arbitrary
auxiliary scores are Gaussian.

## Theory: A Likelihood-Based RL Update

### What This Construction Claims

The goal is an actor/critic data loss that **is** a normalized conditional
negative log likelihood, not a claim that an arbitrary RL loss eventually
becomes one after enough optimization. Advantage-weighted regression provides
the algorithmic starting point: use returns to weight regression onto
experienced actions. Control-as-inference and maximum-a-posteriori policy
optimization provide related probabilistic perspectives. The particular joint
model and derivations below are this plan's construction, not a theorem
attributed to those papers. [AWR][awr] [Control as inference][inference]
[MPO][mpo]

There are three distinct statements to keep separate:

1. The proposed data objective is exactly a likelihood objective for a stated working model.
2. Its action update has a local policy-improvement interpretation under additional assumptions.
3. Classical likelihood efficiency requires still stronger local modeling, dependence, and identifiability assumptions.

Statement 1 does not establish statements 2 or 3 automatically. In particular,
more data does not make a misspecified working model correct. EWC, ridge, and
STM penalties are explicit additions to the data NLL, not hidden likelihood
terms or components of reported predictive NLL.

### Observations, Actions, And Returns

Let $h$ be the recorded information available before an action: camera history,
timestamped language events, executed action history, recurrent context,
elapsed time, and remaining task time. It excludes future rewards and
privileged simulator state. A trainable feature extractor may encode $h$;
the raw history is the conditioning observation, not a frozen learned feature.

Represent the eight action probabilities with seven identifiable logits:

$$
z\in\mathbb R^7,\qquad q(z)=\operatorname{softmax}(z_1,\ldots,z_7,0),
\qquad u(z)=\sum_{i=1}^8 q_i(z)v_i\in\mathbb R^4.
$$

Here $v_i$ is the calibrated primitive control vector for pan, tilt, steering,
and drive. The backend executes $u(z)$, not a sampled categorical primitive.
During training, exploration samples $Z\sim N(\mu_\Theta(h),\Gamma)$ before
this deterministic mapping. $\Gamma$ is fixed after development. Record the
actual sampled $z$, requested vector, executed vector, and behavior density.
Deterministic evaluation uses $z=\mu_\Theta(h)$. Safety intervention is part of
the recorded environment response, not a reason to relabel a requested action
as though it had never occurred.

Seven logits remove the softmax common-offset ambiguity, but the map into
four physical controls is still many-to-one. Retaining the actual sampled
logits makes the declared likelihood well-defined; it does not make all
logit changes physically meaningful. Report control-space changes and the
local sensitivity of the mapping alongside logit-space KL. Treat wasted
regularization in physically redundant directions as a possible limitation,
not evidence of useful retention.

Use observed finite-horizon returns, initially without bootstrapped targets.
Discretize the known bounded return range into $K$ bins with representatives
$g_j$ and observed label $J=j$. The return head predicts
$c_\Theta(j\mid h)$, with value $\sum_j g_j c_\Theta(j\mid h)$.
Categorical return distributions have a distributional-RL precedent; this plan
does not silently inherit C51's Bellman projection or control algorithm.
[Distributional RL][distributional]

Start with $K=21$; check $K=41$ on development artifacts. Derive the range from
the actual reward and horizon, not empirical extrema that silently clip new
returns. Quantization error in a return is at most half a bin width when the
range is respected. Refining bins reduces that error, not critic estimation
error or all RL approximation error.

Record a lagged baseline $b(h)$ at action time, before its outcome is known.
For fixed temperature $\tau>0$, define

$$
w_j(h)=\exp\!\left(\frac{g_j-b(h)}{\tau}\right)>0.
$$

Baseline values, bin edges, temperature, and the positive-definite action
scale $\Sigma_A$ are fixed conditioning data for a likelihood fit. They are
not differentiated through. Do not recompute historical baselines or retired
weights after a model update. A numerical log-weight cap is permissible only
as an explicit, frozen development choice; report its hit rate and qualify
the policy-gradient approximation below if clipping is active.

### One Normalized Joint Model

With $d_z=7$, use

$$
p_\Theta(j,z\mid h)
=c_\Theta(j\mid h)\,
\mathcal N\!\left(z;\mu_\Theta(h),\frac{\Sigma_A}{w_j(h)}\right).
$$

It integrates to one because each conditional Gaussian integrates to one and
$\sum_j c_\Theta(j\mid h)=1$. Its negative log likelihood is

$$
\begin{aligned}
\ell_\Theta(h,j,z)
&=-\log c_\Theta(j\mid h)
+\frac{w_j(h)}2
(z-\mu_\Theta(h))^T\Sigma_A^{-1}(z-\mu_\Theta(h))
+k(h,j),\\
k(h,j)
&=\frac{d_z}2\log(2\pi)+\frac12\log\det\Sigma_A
-\frac{d_z}2\log w_j(h).
\end{aligned}
$$

Thus the actor's weighted quadratic and the critic's categorical cross-entropy
are exactly one joint NLL up to a parameter-independent term. No arbitrary
trainable actor/critic weight is added outside this model. $\Sigma_A$ provides
a declared, development-calibrated relative scale. Shared actor/critic
features do not invalidate normalization.

Interpretation: conditional on an observed history, higher-return actions are
modeled as more concentrated around a preferred command. This is a statistical
factorization, not a causal claim that a future return generated an action.
The joint model is also **not** the Gaussian exploration distribution. They
serve different purposes and need separate names and metadata.

The single-center action model can be inadequate for genuinely multimodal
choices, such as passing an obstacle on either side. Include recent chosen
direction and executed history in $h$, inspect averaged-command failures, and
retain this limitation. Do not introduce a mixture policy mid-confirmation
without a new likelihood, KL implementation, and versioned comparison.

### Why The Actor Step Can Improve Return

For an unrestricted conditional mean and frozen weights, population weighted
regression gives

$$
\mu_{\mathrm{new}}(h)=
\frac{\mathbb E[w(J,h)Z\mid h]}{\mathbb E[w(J,h)\mid h]}.
$$

Suppose data at this history were generated by Gaussian exploration
$Z\sim N(\mu,\Gamma)$, continuation behavior is held fixed, return is bounded,
and weights are not clipped. Expanding in $1/\tau$ gives

$$
\mu_{\mathrm{new}}-\mu
=\frac{\operatorname{Cov}(Z,g_J\mid h)}{\tau}
+O(\tau^{-2}).
$$

For $\mathcal J_h(\mu)=\mathbb E[g_J\mid h,Z\sim N(\mu,\Gamma)]$, the
Gaussian likelihood-ratio identity gives

$$
\operatorname{Cov}(Z,g_J\mid h)=\Gamma\nabla_\mu\mathcal J_h(\mu).
$$

Consequently the leading displacement is a positive-definite preconditioned
ascent direction, with local leading improvement proportional to
$\nabla\mathcal J_h^T\Gamma\nabla\mathcal J_h/\tau$ when a smooth local
expansion applies. This argument concerns the binned return; fine bins
approximate the original bounded return. The baseline cancels from the
conditional weighted mean when it is an action-independent constant at $h$.

This is a local argument for the weighted-regression mechanism, not a global
RL convergence guarantee. Parameter sharing, finite samples, stale behavior
mixtures in replay, finite temperatures, safety saturation, regularization,
and incomplete optimization can all change the result. Log the behavior
version and action coverage. In particular, fitting returns from old policies
does not create a certified current-policy critic for unobserved actions.

The Gaussian-behavior calculation and the working-model correctness
assumption are separate idealizations; do not claim the collected joint data
must simultaneously satisfy both exactly.

### The Single Fisher And A Cheap Exact Predictive KL

Use scores $s=\nabla_\Theta\log p_\Theta$ and loss gradients
$g=\nabla_\Theta\ell=-s$. The intended Fisher remains the model-induced
quantity $\mathcal I(\Theta)=\mathbb E_\Theta[ss^T]$. At a fixed recorded
conditioning history, write its integrand $F_h$ as

$$
F_h(\Theta)=
\sum_j c_j\,\nabla\log c_j\nabla\log c_j^T
+\overline w_\Theta(h)\,J_\mu^T\Sigma_A^{-1}J_\mu,
\qquad
\overline w_\Theta(h)=\sum_j c_jw_j.
$$

The actor score has zero conditional mean at each $j$, so its cross terms
with the categorical score vanish under the model, including when their
parameters overlap. LTM and STM Fisher blocks are coordinate blocks of this
one matrix, not different Fisher estimands. History sampling approximates the
conditioning average. Changes in visitation, behavior, or a weight recipe
are not derivatives with respect to $\Theta$ at fixed history.

For two parameter values sharing the same conditioning design,

$$
\operatorname{KL}(p_\Theta\Vert p_{\Theta'})
=\operatorname{KL}(c_\Theta\Vert c_{\Theta'})
+\frac{\overline w_\Theta(h)}2
\|\mu_\Theta(h)-\mu_{\Theta'}(h)\|_{\Sigma_A^{-1}}^2.
$$

This is exact for the declared joint model. It needs neither a dense Fisher
matrix nor action samples. The reverse KL uses the other categorical
distribution and its corresponding $\overline w$. Use this formula for STM
finite differences, and test it against numerical/model-sampling references
on a tiny network before using the VLM.

For the eviction Fisher diagonal, a small model can sum output-Jacobian
contributions exactly. Production may use model-sampled per-record scores:
sample $j$ from $c_\Theta$, then $z$ from its conditional Gaussian, detach both
synthetic labels, and differentiate their log likelihood. Average squared
per-record scores, never the square of a batch-mean score. Compare one, two,
and four model-label draws in a bounded pilot. Use the same estimator in
both conditions. Do not replace the Fisher by squared gradients of the
regularized training objective or by an unlabeled empirical-Fisher shortcut.

No full eigendecomposition, parameter Hessian, or dense Fisher is required.
If an implementation still needs one, stop and reconcile that change with
these formulas before running it.

### Statistical Scope, Dependence, And Sample Counting

The likelihood framing permits classical local reasoning when the working
model is locally adequate, the relevant directions are identifiable,
conditional scores are suitably centered, moments exist, dependence is
controlled, and optimization approximates the intended solution. EWC and
STM penalties then define a regularized likelihood problem; they do not
magically preserve unregularized MLE efficiency.

LoRA factor gauges and redundant physical effects can make parameter Fisher
matrices singular. Fix the factor scaling convention, avoid inverse-Fisher
claims on the full raw parameterization, and evaluate functional predictions.
Do not claim a LAN approximation merely because external drift is smooth;
measure changes in predictive distributions and update adequacy.

An individual transition is a legitimate observation even when adjacent
transitions are dependent, just as observations in a time-series likelihood
remain observations. It is not automatically an independent observation.
Keep these counts separate:

- **New experience:** unique acquired transitions, counted once.
- **Replay exposure:** repeated uses of those transitions in optimization.
- **Sequence context:** supporting frames/actions needed to interpret a transition.
- **Independent uncertainty units:** fresh paired learning replicas, not replay draws or video frames.

Report return/score autocorrelation and an explicitly diagnostic correlation
length if estimable. Do not advertise an iid effective sample size obtained
by counting replay draws. Formal inference uses paired replica outcomes;
episode dependence remains inside each replica.

A sandwich covariance is not part of the initial implementation. Normalized
likelihoods do not remove misspecification, but adding a difficult covariance
estimator is not required for the primary applied comparison. State the
model-based Fisher interpretation and use replica-level uncertainty. Revisit
robust covariance only if a specific calibration failure makes it useful.

## STM, LTM, And Retirement

### Parameterization

For each selected transformer linear map, use parallel additive adapters:

$$
W_{\mathrm{eff}}=W_0+s_LB_LA_L+s_SB_SA_S.
$$

$W_0$ is frozen and quantized where supported. The LTM and STM factors are
separate trainable banks. This is not LoRA recursively applied to another
adapter. Keep factor scales, initialization conventions, placement, and
parameter ordering fixed after development. Do not merge, rebalance, or reset
adapters in the middle of a trajectory.

The small LSTM's dense recurrent and readout parameters are LTM. Add STM
low-rank contributions to declared gate/projection/readout matrices, with
tests against the equivalent dense effective matrices. The initial candidate
is rank-one STM and rank-four LTM transformer adapters; profile alternatives
as described below. Both conditions have every declared trainable coordinate.

All ordinary likelihood gradients from replay may update both banks. EWC
protects only LTM. STM FD gradients may update every parameter on which that
penalty actually depends, including LTM; do not detach LTM merely because
the finite-difference displacement is in STM coordinates.

Two parallel low-rank banks can represent more than the LTM bank alone. This
is controlled by the matched-capacity comparison, not denied by terminology.
Lower STM sensitivity is a diagnostic, not proof that information has moved
into a uniquely identifiable LTM representation. [LoRA][lora]

### Objective And Age-Dependent FD

At an optimization event, let $B$ contain newly labeled, not previously fitted
transitions and $R$ the eligible replay sample. Do not include a record in both
roles in that event. Use equal weights on their mean NLLs when both exist;
use the available mean when only one exists. Sleep uses the replay mean.
Record these cases explicitly because returns arrive after actions.

With LTM anchor $a_t$, archived diagonal $D_t$, and fixed adaptation weight
$\pi_0$, set $\omega=(1-\pi_0)/\pi_0$ and optimize

$$
\mathcal L_t(\Theta)=\mathcal L_{\mathrm{data},t}
+\frac{\omega}{2}(\Theta_L-a_t)^TD_t(\Theta_L-a_t)
+\mathcal R_t(\Theta)+\lambda_{\mathrm{FD}}\mathcal C_t(\Theta).
$$

Use the existing scale-normalized ridge convention explicitly:
$s_t=\operatorname{tr}(D_t)/d_L$, $\kappa_t=c_r\max(s_t,s_{\min})$, and

$$
\mathcal R_t=\frac{\omega\kappa_t}{2}
\left(\|\Theta_L-a_t\|^2+
\|\Theta_S-\Theta_{S,t}^{\mathrm{preupdate}}\|^2\right).
$$

Record the numerical floor $s_{\min}$ separately from statistical ridge.
Freeze the pre-update anchor across all optimizer iterations of that event.
No implicit AdamW decay may duplicate this penalty. Ridge and all objective
weights are visible in the recipe and notebook settings.

For replay item $i$, let $a_i\in[0,1]$ be its clipped age since arrival divided
by replay capacity, measured in newly acquired transitions. Let $E_Sv$ embed
a probe into STM coordinates, with $\mathbb E[vv^T]=I_{d_S}$. Define

$$
\Theta^\pm=\Theta\pm\frac{\epsilon}{2}E_Sv,
\qquad
\Phi_i(\Theta,v)=
\frac{\operatorname{KL}(p_{\Theta^+}\Vert p_{\Theta^-})+
\operatorname{KL}(p_{\Theta^-}\Vert p_{\Theta^+})}{\epsilon^2},
$$

$$
\mathcal C_t=\frac{1}{|R|d_S}\sum_{i\in R}a_i^2
\frac1P\sum_{p=1}^P\Phi_i(\Theta,v_p).
$$

The histories and weight recipes are the same at both endpoints. For smooth
models, the probe expectation approaches
$\operatorname{tr}(F_{h_i,SS})/d_S$ as $\epsilon\to0$. This is sensitivity in
the declared fixed adapter coordinates, not a gauge-invariant count of stored
observations. Use functional STM perturbation and prediction diagnostics too.
Freeze probes within an optimizer event; draw them from a named RNG stream.
Validate half/double step sizes, quantization resolution, and both-endpoint
autograd. Do not silently substitute directional KL or a detached endpoint.

### What Enters EWC

An online observation enters the LTM archive **once, on retirement from
eligible replay**, after its return label and last permitted replay use.
Neither repeated replay, reward completion, nor sleep creates a second
admission. Initialization has a separate ledger distinguishing observations
already retired from those still in the initial buffer.

Start with direct diagonal EMA, not LFU transport. With $e$ newly retired
items and nominal arrival unit $m_0=1$ transition,

$$
\eta_e=1-(1-\pi_0)^{e/m_0},\qquad
D_{t+1}=(1-\eta_e)D_t+\eta_e\widehat F_{LL,\mathrm{evicted}},
\qquad a_{t+1}=\Theta_L^{\mathrm{accepted}}.
$$

Do not recenter or update the archive if $e=0$. Store the resolved aggregation
rule when several retirements occur together. Evaluate their Fisher at the
accepted parameter value and their frozen conditioning recipes. This is an
online-EWC approximation, not the exact old-data likelihood Hessian or an
exact sufficient statistic. The fixed starting choice is $\pi_0=.025$;
development may choose a common alternative, never an adaptive rule in this
study. [EWC][ewc]

Keep the original parameter-learning process distinct from maintenance of
the Fisher summary. This plan explicitly omits LFUs; it does not implement
only part of $C+R$ and call it a full LFU.

## Shared Robot Interface And Deployment

### Small Interface Below The Learner

Define backend-neutral observation, command, and receipt schemas. Implement
the same contract for a simulator and a physical PiCar client. Required fields
include schema version, run/session ID, monotone command sequence ID, capture
and receipt timestamps, observation ID, bounded four-control vector, command
duration/expiry, executed controls, and safety status. Simulation-only reset
and privileged diagnostic state belong in a separate administrative interface.

Use explicit units and semantics. Camera absolute angles versus increments,
steering angle versus normalized command, drive speed versus drive duration,
and neutral values must not depend on the backend. The old controller's
nonzero-steering/zero-drive side effect must not silently move the new robot.
Resolve it in a calibrated mapping and version the intentional behavior change.
Do not assume zero is a safe neutral value for an absolute camera servo.

Preserve the existing eight primitives and probability-weighted mapping from
`src/picar_kl/actions.py`; port it with a golden mapping test, then apply one
explicit physical-unit calibration. Applying a mixture and sampling a
primitive are different experiments. Validate forward, reverse, stop, camera,
and steering combinations, including opposing actions that cancel.

The legacy normalized vectors, before physical-unit calibration, are:

| Primitive | Pan | Tilt | Steering | Drive |
|---|---:|---:|---:|---:|
| `drive-left` | 0 | 0 | -1 | 1 |
| `drive-right` | 0 | 0 | 1 | 1 |
| `drive-forward` | 0 | 0 | 0 | 1 |
| `drive-backward` | 0 | 0 | 0 | -1 |
| `look-left` | 1 | 0 | 0 | 0 |
| `look-right` | -1 | 0 | 0 | 0 |
| `look-up` | 0 | 1 | 0 | 0 |
| `look-forward` | 0 | 0 | 0 | 0 |

The last primitive provides a nominal neutral command; hardware watchdog
stopping bypasses the learned distribution entirely. This table does not
establish servo signs or prove that a weighted neutral camera value has the
correct physical meaning. Measure those properties in the interface audit.

The learner never imports a simulator-specific pose or controller. Future
physical demos should replace the backend and calibration, not rewrite the
learning algorithm. Match latency, camera geometry, action saturation, steering
response, braking, and frame timing using user-supplied measurements when
available. Until then, label the simulator parameters as unvalidated defaults.

### Raspberry Pi Responsibilities

Maintain a lightweight robot-side service under `demo/picar/`, with its own
minimal dependency set and deployment instructions. Torch, Qwen, training,
and artifact analysis stay off the Pi. Port appropriate parts of
`src/picar_kl/robot/server.py` and the existing hardware controller after
checking license/provenance; do not import the external checkout at runtime.

The Pi must enforce bounds, command expiry, local watchdog stopping, and
session/sequence deduplication. Retrying an acknowledged command must not
repeat its motion. Reconnection stays disarmed until explicitly armed; a
new session cannot replay stale queued commands. Use authenticated,
non-public control endpoints and POST for state-changing requests, with
bounded payloads and timeouts. Do not expose an unauthenticated motor service.

Physical calibration or motor operation always requires a separate explicit
user action. Automated tests use a fake driver. A verbal task instruction is
not the emergency-stop mechanism; the local watchdog and operator stop remain
independent of model inference. No downloaded model or simulator setup script
may trigger physical motion.

### Existing Code To Reuse Carefully

| Source in `kl-projection` | Useful starting point | Required correction or check |
|---|---|---|
| `src/picar_kl/actions.py` | Eight primitives and weighted four-vector | Unit-calibrated backend semantics |
| `src/picar_kl/models/vlm_head.py` | Transformer conditioning head | Use language-attending features in the live path |
| `src/picar_kl/models/phase2.py` | Student construction | Current prefix head is not the required language path |
| `src/picar_kl/models/lstm_policy.py` | Small recurrent policy | Explicit recurrent-state and model-version handling |
| `src/picar_kl/context/state.py` | Language/action history | Executed actions and instruction completion, not repeated commands |
| `src/picar_kl/reward/prompts.py`, `reward/scoring.py` | Explicit graded visual reward and generated numeric JSON | Reproduce RGB scoring before adapting the rubric; completion-token probability is a different quantity |
| `src/picar_kl/context/messages.py`, `vlm/control.py` | System/user roles, operator corrections, action/status messages | Port causal semantics into this repository; preserve the executed weighted control |
| `src/picar_kl/io/observation_store.py` | Per-frame storage and metadata | Checksums, deduplication, bounded caches, archive eligibility |
| `src/picar_kl/legacy/robot_demo/src/model/replay_buffer.py` | Transition replay semantics | Replace bulky tuple duplication and unbounded serialization |
| `src/picar_kl/legacy/robot_demo/src/model/backbones.py` | Quantized Qwen loading | Top-$L$ restriction and verified gradient boundary |
| `src/picar_kl/robot/server.py` | Lightweight injectable service | Watchdog, session/sequence protocol, authentication |

Reuse coherent code, not hidden assumptions from the legacy composite RL loss.
Record copied-file provenance and licenses. Do not restructure historical
MNIST implementations just to make their names generic.

## Model, Language, And Timing

### Shared Backbone, Separated Roles

Begin with the existing Qwen2.5-VL-3B-Instruct family, pinning an exact model
revision, processor, tokenizer, quantization setup, and software environment.
One frozen quantized backbone is shared across teacher/evaluator and learner
roles. Explicit role-specific adapters and heads isolate the frozen
teacher/reward semantics from online learning. No implicit switch may leave
learner adapters active during reward scoring. [Qwen model card][qwen]

**Reproducibility amendment (2026-11-09):** the explicit model-loading command
downloads the pinned revision into this project's ignored model cache by
default, then reuses it. `--model-path` optionally points to an existing model;
it is not required for a fresh clone. `--offline` must fail clearly on missing
assets instead of downloading or silently changing models. No import or
reporting operation downloads a model. Software dependencies must be installed
from declared packages or provided by this repository, never imported from a
sibling checkout or personal absolute path.

Freeze the vision tower and lower transformer blocks. Restrict trainable LoRA
to the top $L$ selected transformer blocks and explicitly listed heads.
Profile $L\in\{1,2,4\}$ sequentially, not as a factorial search. PEFT supports
layer selection, but configuration alone is not proof of a short backward
graph. Verify `requires_grad`, saved tensors, and peak allocated/reserved
memory. Run the frozen prefix without autograd and detach its outputs only
where all trainable dependencies are truly above that boundary. Do not
accidentally freeze a trainable language or vision projection. [QLoRA][qlora]
[PEFT layer selection][peft] [PyTorch autograd][autograd]

Training the top blocks still costs forward passes through the frozen prefix.
Measure actual image-token and text-context lengths after preprocessing.
Caches are valid only for identical inputs, prefix weights, processor settings,
and relevant positional/context information; a frozen backbone does not make
arbitrary history features reusable. Separate teacher/evaluator KV caches
from learner caches and invalidate trainable-role caches on publication.

The LSTM receives visual/action history plus a conditioning vector from a
transformer computation that actually attended the current language context.
No generated motor-control text is needed after distillation. Language events
remain in context alongside their subsequent executed actions and elapsed
time, so "turn right" can become a completed action rather than a perpetual
fresh command. Bound and version context truncation; do not silently discard
an unfinished instruction. A new instruction or materially changed history
can require a new feature computation, whose cost must be counted.

Test identical visual histories with contrasting instructions, and test the
same instruction before and after its requested action has been completed.
Log feature age, context contents, and command changes. Neither a class named
"conditioning head" nor a transformer hidden state alone proves correct
language wiring.

### Online Scheduling And Sleep

Use separate physics/capture, inference, and learning workers with bounded
queues. Control inference has GPU priority; training starts only in available
windows, publishes parameters atomically, and never modifies a model during
an in-flight inference. Serialize role execution on the shared GPU unless
profiling demonstrates safe overlap. Reward scoring can be deferred, but its
backlog and eventual compute are charged.

Physics continues while inference or training runs. At each nominal control
deadline, use the newest valid command or the declared safe fallback; discard
expired commands. Do not accelerate catch-up steps to hide a stalled policy.
Log capture-to-command latency, inference time, update time, batch size,
missed deadlines, fallback actions, and model age. The default simulator runs
at real-time speed; accelerated development diagnostics are labeled separately.

Learning occurs between action decisions when capacity permits, not by
freezing a moving world. If real-time operation is infeasible, first reduce
trainable layers, context, resolution, or update work through the common
profiling policy. A slower control period is allowed during development only
after checking task controllability and recording the change for both arms.

Use periodic stationary "sleep" intervals for larger replay batches and more
optimizer work. Default eligible data is the current replay buffer only.
Freeze the sleep schedule before confirmation, initially one interval per
approximately 400 new transitions at a safe episode boundary. Count sleep
compute and time unavailable for interaction separately. Sleep and test
episodes do not advance the scripted training drift clock, by explicit
experimental design; this is not a claim that natural drift stops while a
physical robot charges. During a declared sleep interval the controller is
stopped, not pretending to meet active-task deadlines.

Reusing already retired observations during sleep is a required future,
default-off option. **Implementation status:** `LearningConfig.validate`
currently rejects `retired_archive_replay=True`; this option has not yet been
implemented. Finish its explicit evidence-accounting protocol before claiming
support. Enabling it is a separately versioned experimental
condition, requiring an explicit accounting rule for repeated evidence and
archived EWC; it is not an autonomous performance rescue in the primary pair.
The later Phase 1 Fix 2, item 5, now assigns implementation and acceptance of
that default-off option; this paragraph records its pre-Fix-2 status.

## Simulator And Continuous Drift

### Task And Geometry

**Revision notice:** the specification below defines the original
`safe-stop-v1` task used by the completed readiness artifacts. The pending
Phase 1 Fix 2 proposes `ball-contact-v1` for future acquisition:
target contact is success; obstacle/wall contact is failure. Its diagnostic
ladder and precedence rules are specified there. Do not relabel earlier
collisions as successes or silently combine task versions. The obstacle,
action interface and continuous drift remain part of the proposed full task.

Use PyBullet for dynamics and Gymnasium-compatible episode conventions,
rather than a hand-written kinematic game engine. The robot must navigate
around an obstacle, approach a target, and stop at an acceptable distance.
Keep graphics and sensors simple enough to run alongside the VLM. [PyBullet][bullet]

The following are engineering defaults from the handoff, not measurements of
the physical PiCar:

| Item | Initial value |
|---|---|
| Arena | $3.6\times2.4$ m; walls 0.35 m high |
| Vehicle footprint / wheelbase | $0.32\times0.18$ m / 0.20 m |
| Nominal start | $(0.5,0)$ m, facing the target |
| Obstacle | Center $(1.575,0)$ m; $0.25\times0.50\times0.09$ m |
| Target | Radius 0.10 m; center $(2.9,0,0.10)$ m |
| Camera | Relative position $(0.06,0,0.18)$ m; pitch $-8^\circ$ |
| Camera geometry | Horizontal FOV $90^\circ$; 4:3 aspect ratio |
| Render / candidate learner image | $448\times336$ / $160\times120$ or $224\times168$ |
| Forward / reverse speed limit | 0.20 / 0.10 m/s |
| Steering bound | $\pm25^\circ$ before drift calibration |
| Physics step / nominal control interval | $1/240$ s / 0.5 s |
| Episode deadline | 20 physical seconds |

Derive vertical FOV from horizontal FOV and aspect ratio rather than setting
both to $90^\circ$. Simulate bounded camera pan/tilt because the accepted
action mapping includes them. A fixed-camera diagnostic must explicitly mask
the same controls in all compared methods; do not quietly remove the camera
channels from only the simulator.

Start-position jitter is initially $x\in[0.4,0.6]$ m,
$y\in[-0.15,0.15]$ m, and yaw within $\pm10^\circ$; target lateral jitter
is $\pm0.2$ m and obstacle lateral jitter $\pm0.15$ m. Include mirrored layouts.
Check both passages against vehicle turning radius and camera visibility.
Reject impossible resets using a method-independent rule fixed before fresh
evaluation. The privileged scripted controller is a feasibility diagnostic,
not an unreported source of primary demonstrations.

Success requires front-bumper clearance to the target in $[0.25,0.45]$ m,
heading error at most $15^\circ$, speed at most 0.02 m/s, and one continuous
second satisfying these conditions without contact. Define geometric surfaces
precisely in tests. Collision takes precedence over success; a valid success
at the final allowed physics tick precedes deadline failure. Record failure
type, clearance, heading, and completion time as well as the binary result.

The learner and Qwen teacher see RGB, language, their available history, and
timing. They do not see world pose, target coordinates, steering gain, drift
index, or a top-down map. Privileged geometry is allowed for terminal labels,
feasibility checks, and reporting, with that hybrid supervision stated plainly.

### Smooth Outward-And-Return Change

Let $\alpha\in[0,1]$ blend A and B. Start with an A floor texture and steering
gain 1; B uses a visibly different floor texture and steering gain 0.65.
Interpolate texture intensities and use gain $1-0.35\alpha$. Preserve geometry
and feasibility. Do not add more drift variables until these two have passed
the acquisition/readiness checks.

For normalized ramp progress $r\in[0,1]$, use
$q(r)=10r^3-15r^4+6r^5$. The training exposure schedule is:

| Fraction of training exposure | Regime |
|---|---|
| 0--10% | Stable A |
| 10--45% | Smooth A-to-B ramp |
| 45--55% | Stable B |
| 55--90% | Smooth B-to-A ramp |
| 90--100% | Stable A |

Interpolate within physics steps; do not jump to a new texture/gain at each
episode reset. Pilot at least 200 new transitions per ramp and several
episodes, then inspect predictive change per observation and tracking lag.
Continuity alone does not certify that change is slow enough for a local
statistical approximation. Conversely, a visible texture change alone does
not establish a learning challenge.

Use the same exogenous drift schedule at matched training exposure counts.
Within each nominal acquisition interval, interpolate its fractional progress
using physical time. Count captured intervals even when inference misses its
deadline and a fallback command executes. Pauses, sleep, resets, and held-out
evaluation are explicitly outside this training exposure clock. Log physical
time, simulator time, wall time, and acquisition count separately.

### Reward And Return Labels

**Revision notice:** the equations immediately below describe the retained
`potential-v1` reward used in earlier implementation/diagnostics. The pending
Phase 1 Fix 2 defines `dense-visual-v1` for new runs, together with
revised return bounds and a bin-resolution audit. Changing only the evaluator
prompt would not implement that amendment. No scientific learning results
have yet been produced under either reward version.

A frozen evaluator scores a potential $\Phi(h)\in[0,1]$ from a short RGB
history with timestamps and a fixed prompt. It must not read privileged state
or future frames. Cache each endpoint potential once per model/prompt/history
version and reuse it consistently in neighboring rewards.

Use a time-consistent discount
$\gamma_t=\gamma_0^{\Delta t_t/\Delta t_0}$, with initial $\gamma_0=.99$ and
$\Delta t_0=0.5$ s. The proposed reward is

$$
r_t=\gamma_t\Phi(h_{t+1})-\Phi(h_t)
-.02\,\Delta t_t/\Delta t_0+r_t^{\mathrm{terminal}},
$$

with terminal bonuses $+10$ for success, $-5$ for collision, and $-1$ for
deadline failure. Set terminal potential to zero. The actual elapsed-time
discount and step cost are deliberate refinements of a fixed-step handoff
reward; freeze them for all methods. Derive conservative return bounds from
these quantities and the finite horizon. Record each component separately.

Potential-based shaping has a policy-invariance rationale under its stated
conditions. In this complete-return formulation it also telescopes:

$$
G_t^{\mathrm{shaped}}=G_t^{\mathrm{base}}-\Phi(h_t)
$$

when the terminal potential is zero and the same discount products are used.
Therefore do not promise that VLM shaping automatically solves sparse-return
credit assignment here. It offers diagnostics and a possible later
bootstrapping ingredient; complete-return AWR can cancel its conditional
baseline effect. Test this identity rather than presenting shaped reward as
extra independent supervision. [Potential shaping][shaping]

The deadline is a real finite-horizon task ending, with remaining time exposed
to the controller. Infrastructure interruptions are not deadline failures,
collisions, or zero returns. A partial interrupted episode must resume
correctly or remain explicitly incomplete. Distinguish termination from
truncation in both stored data and return calculation. [Time limits][time]

The evaluator's outputs must satisfy a strict finite numeric schema. Retry
invalid scoring offline before finalizing labels; never turn parser failure
into reward zero or selectively discard bad-task outcomes. Report failures,
label delay, queued work, and scoring cost. This remains a hybrid simulator
reward, not demonstrated VLM-only reward learning on a physical robot.

## Initial Collection, Replay, And Storage

### Qwen Collection And One-Time Distillation

Collect A-domain experience with Qwen controlling the robot and no online
parameter learning. Use timestamped natural-language task instructions and
coaching, including recovery, stopping, and completion cases. Autonomous
runs use a fixed, visible-history-based coaching script; user participation
is optional and never a dependency for finishing the simulator study.
Human-assisted data, if added, is tagged and shared across conditions.

For the pending Fix 2 revision, bind this collection to the newly admitted
task/reward/coaching identities. Reuse neither a historical completion-score
dataset nor a legacy critic merely because its dimensions match. Publish
assisted and unassisted readiness separately. The new dense visual rewards
must survive return discretization before fitting the shared initial model.

Start with 300--600 transitions and a provisional ceiling of 1,200 before
reforecasting. Store the teacher's eight-action distribution and its actually
executed weighted control. Specify a fixed validated extraction method from
the teacher output. If only hard commands are available, label them as hard
demonstrations; do not fabricate calibrated teacher probabilities.

Teacher distributions can contain exact zeros, which have no finite reference
logits. Before teacher execution, apply a declared small probability floor
and renormalization, record both distributions, and compute
$z_i=\log(q_i/q_8)$ from the executed distribution. Use this same distribution
for the distillation target and the requested weighted control. Calibrate
the floor's physical effect, especially on stopping, and freeze it before
collection. Do not smooth only after execution and then pretend those logits
generated the observation. A smoothed hard instruction remains an engineered
demonstration target, not an estimate of teacher uncertainty.

Initialize the student with
$\operatorname{KL}(q_{\mathrm{teacher}}\Vert q_{\mathrm{student}})$ on
eligible recorded histories. This is the one-time KL projection. Then perform
a bounded A-only fit of the declared joint return/action likelihood so the
online critic and score model are initialized before building EWC. Distillation
and the joint likelihood are different objectives; record their stages rather
than mixing their gradients into an undocumented Fisher. Freeze a common
initialized checkpoint, return-bin design, and initial replay/archive ledger.
[Knowledge distillation][distillation]

After this point the LSTM emits all ordinary motor actions. Qwen remains the
shared contextual backbone and frozen evaluator, not a continuing stream of
teacher action labels. New learning uses observed experience and the stated
RL likelihood. Progress & Compress is a useful knowledge-consolidation
precedent, but this plan does not copy its repeated distillation schedule.
[Progress & Compress][progress]

Use one shared bootstrap asset for feasibility unless profiling supports
independent initializations. Fresh replicas then measure uncertainty
**conditional on that common bootstrap**, not across independently pretrained
teachers. Count bootstrap data and compute separately and also show end-to-end
totals; online savings must not conceal a large startup cost.

### Transition Store And Recurrent Context

Store each frame once with content/checksum references. A transition record
contains current/next frame IDs, timestamps, instruction/context IDs, requested
logits and action distribution, requested/executed controls, receipts, reward
components, return/bin, lagged baseline, behavior/model version, and terminal
or interruption flags. Keep return-label readiness separate from acquisition.

Finite-horizon returns become available at episode end. Maintain a bounded
pending episode store in addition to the replay buffer; count it against
working memory. Schedule fitting of newly labeled transitions without
pretending they are new acquisitions at label-completion time. Every new
transition gets a fixed optimization allowance; delayed labels do not create
an unlimited burst of updates that pauses physics.

Sample individual training transitions with a short reconstructible history,
not whole episodes as the sampling unit. History frames and actions still
occupy storage and may overlap strongly. Count their bytes and references in
the replay budget. Do not retrieve retired observations for free recurrent
burn-in. Resolve boundary context explicitly: initially use a fixed context
length, masked at real episode starts, and a declared policy for context
whose predecessors have left eligibility.

Prefer reconstructing recurrent state under current weights over treating an
old stored hidden state as current. Parameter updates can make hidden states
stale; recurrent replay literature documents this practical issue. On actor
publication, rebuild live state from the bounded retained history, or publish
only at a safe episode boundary if the rebuild cannot meet deadlines.
Checkpoint this policy and all required context. [Recurrent replay][r2d2]

Start with FIFO capacity $C=128$ transitions and replay batch 4. Profile
capacities 64/128/256 and batches 1/4/8 only to establish a feasible common
envelope, not to start another buffer-allocation study. A context dependency
that extends effective retention must be disclosed and bounded. Keep archival
raw data physically available for audit but logically ineligible for training
after retirement.

### External Archive And Quotas

Use a configurable `--archive-root` for large immutable raw data, video,
completed checkpoints, and manifests. Keep the active working set, indexes,
and bounded decode caches on the primary disk. Never publish a user-specific
mount path in source, configs, or example commands.

Proposed initial quotas are 100 GB local working artifacts and 1 TB external
artifacts, subject to actual free-space checks and the user's archive contents.
The presence of an 8 TB drive is not permission to consume all free space.
Keep at least the greater of 20 GB or 10% of available allocation as reserve;
reforecast bytes per transition, model checkpoint, and video in Phase 1.
Include temporary files, failed runs, and archive copies in admission checks.

Prioritize compact transitions, reproducibility metadata, and learner states
over dense full-time video. Reconstruct or selectively render video from
recorded simulator states/actions where validated; raw camera samples needed
by replay remain exact. Store the frozen backbone once by revision/checksum;
do not duplicate its billions of parameters in every trajectory checkpoint.

If the external drive disappears, stop admitting new work, preserve a bounded
local checkpoint, and report the reason. Do not silently spill terabytes onto
the primary disk. Verify checksums before pruning a copied local artifact.
Never move or delete code as part of archive management.

## Applied Metrics And Statistical Efficiency

### Primary Story

Report **new-task capability, old-task retention, samples, and compute together**.
Use held-out reset/layout panels that never enter training, replay, EWC,
reward calibration, or hyperparameter selection after the development split.
Frozen A and B panels hold the environment fixed across checkpoints; a
current-regime panel is useful diagnostically but cannot substitute for them.

All success/collision metrics carry the admitted task version. Under the
proposed `ball-contact-v1`, target contact is successful acquisition, while
wall/obstacle contact is an unsafe collision. Do not use the old aggregate
contact-as-failure statistic or pool it with `safe-stop-v1` outcomes. Show
visual reward separately from this held-out operational capability.

Let $p_m^B(n)$ be B-panel task success after $n$ unique new transitions under
method $m$, and $F_m^B(n)=1-p_m^B(n)$. Define budget-averaged failure

$$
A_m^B(N)=\frac1N\int_0^N F_m^B(n)\,dn,
$$

using the declared interpolation between measured checkpoints. The primary
paired effect is $A_{\mathrm{control}}^B-A_{\mathrm{FD}}^B$: positive means
better new-task performance across the same experience budgets. This does
not require fitting a linear learning curve. Show actual checkpoints and
uncertainty, not only a smoothed area statistic.

For A retention, use
$H_m^A(n)=F_m^A(n)-F_{\mathrm{initial}}^A$ and report both budget average and
endpoint, plus the FD-minus-control gap. Also report collision rate, task
completion time, final clearance/heading error, and instruction-following
checks. Continuous geometric diagnostics explain near misses but do not
replace the operational success definition after seeing results.

Fresh B tests assess genuinely new performance. A tests assess retained
capability after its supporting training records have aged out of replay.
Additionally, a small fixed retired-cohort panel may report predictive NLL and
STM perturbation sensitivity, clearly separated from live task success and
never used for optimization. Fisher trace is a mechanism diagnostic, not a
claim that an observation resides in one parameter bank.

### Samples Saved And The S Statistic

Use the outward ramp and B hold, ending at 55% of exposure, for sample
equivalence. A whole outward-and-return cycle need not have monotone learning
curves, so do not fit one line through it or claim a global sample multiplier.

The proposed capability target is 70% B success; development may select one
other reachable target before fresh runs, with the reason recorded. Define
the primary target crossing as the first of two consecutive measured
checkpoints meeting the target and the declared A-retention ceiling. Show
attainment fractions and interval/censoring information. If the panel is too
small to estimate crossings usefully, report budget-averaged performance and
label sample equivalence unavailable rather than manufacture precision.

For supported interpolated mean-curve crossings $n_m(q)$ at the same capability
and acceptable retention, report

$$
f(q)=1-\frac{n_{\mathrm{FD}}(q)}{n_{\mathrm{control}}(q)},\qquad
c(q)=\frac{T_{\mathrm{FD}}(n_{\mathrm{FD}}(q))}
{T_{\mathrm{control}}(n_{\mathrm{control}}(q))},\qquad
S(q)=\frac{f(q)}{c(q)}.
$$

**Elevator pitch:** $f$ is the fraction of new experience saved. $S$ divides
that saving by the compute multiplier needed to reach the same capability.
For example, saving half the experience at twice the compute gives $S=.25$.
Neither $S$ nor a fitted crossing is itself a probability of success.

$T$ includes operational inference, learning, reward labeling, Fisher
maintenance, and sleep through the crossing. Report learner-only compute too,
but do not substitute it for operational cost without changing the label.
The complete study ledger additionally includes initialization, held-out tests,
profiling, failures, and reporting. Show online-only and bootstrap-inclusive
sample costs separately. Different cost definitions must not masquerade as
directly comparable historical $S$ scores.

Interpolate only within observed brackets. No extrapolation to targets never
reached, no automatic linear fit, and no silent exclusion of bootstrap draws
that fail to cross. Display the supported-draw fraction and censoring. A plot
of both success curves with a horizontal target and horizontal sample-saving
arrow is the main explanation; a compact table supplies uncertainty and cost.

### Pairing And Uncertainty

Use six fresh paired replicas as the initial target. Each pair shares the
bootstrap checkpoint, exogenous reset panel, drift, sensor-noise stream,
coaching policy, and named exploration random variates where meaningful.
Different actions necessarily produce different trajectories; do not claim
identical experienced states or rewards. Evaluation uses common held-out
layouts and sensor noise, with no parameter updates.

Compute paired mean effects and 95% intervals across complete replicas.
Use whole-pair bootstrap resampling for nonlinear sample-equivalence
summaries, retaining failed-crossing information. Six replicas are a modest
uncertainty sample; treat mechanism and multiple secondary comparisons as
exploratory. Do not use individual transitions, evaluation frames, or replay
draws as independent replicas. Report all planned pairs, including poor runs.

No significance-driven stopping. Operational pauses and budget cutoffs are
allowed, with progress and incomplete-pair counts visible. A retention-gap
interval containing zero does not establish equality; show its width against
the declared practical tolerance.

## Experimental Phases

### Phase 0: Model-Split Implementation And Acceptance

**Purpose:** leave Sol a tested, executable experiment, not a collection of
mathematical sketches that require the user to switch models while away.
Reserve Astra's effort for translating the mathematical contract into correct
numerical software; delegate the substantial surrounding engineering to Sol.

Implement the simulator, shared protocol, lightweight Pi service with fake
hardware tests, top-$L$ model/adapter wiring, one-time distillation path,
joint likelihood, exact KL, diagonal Fisher, retirement ledger, delayed
returns, replay/context storage, scheduler, reporting, and resume mechanics.
Keep modules small and reuse established project artifact/seeding utilities
where appropriate. Document numerical conventions next to the implementation.

#### Model Routing And Ownership

Use these explicit agent settings, verified against the delegation interface
when this amendment was written on 2026-11-09:

| Requested model | Model identifier | Reasoning effort |
|---|---|---|
| Astra 6 Extra High | `gpt-6-astra` | `xhigh` |
| Sol 6.1 Extra High | `gpt-6.1-sol` | `xhigh` |

Set both fields on delegated assignments; do not rely on parent defaults or
substitute an older Sol model. These settings choose sub-agents, not the model
of an already-running parent conversation. Prefer a Sol parent for routine
coordination; if the parent is Astra, still delegate the engineering work and
avoid duplicative supervisory analysis. If an explicitly requested model is
unavailable at execution time, report that limitation rather than silently
substituting a different model or reasoning effort.

Official [model identifiers][openai-models] and
[model-selection guidance][openai-selection] are supporting references;
availability of these overrides in the actual delegation tool is the
operational check. This amendment changes work allocation, not experimental
scope, numerical definitions, or the evidence required for acceptance.

| Sub-phase | Owner | Deliverable and boundary |
|---|---|---|
| 0A: Numerical contracts | Astra 6 Extra High | Typed numerical inputs/outputs, detach/freeze rules, state-transition ordering, and tiny reference fixtures for integration |
| 0B: Numerical implementation | Astra 6 Extra High | Joint NLL, exact KL, diagonal Fisher, STM FD, EWC/ridge objective, return/weight calculations, and mathematical unit tests |
| 0C: Model and language plumbing | Sol 6.1 Extra High | Quantized loading, top-$L$ adapters, role isolation, recurrent model, context/cache lifecycle, and one-time distillation wiring |
| 0D: Simulator and robot interface | Sol 6.1 Extra High | Shared protocol, dynamics/rendering, drift, terminal events, Pi service, fake hardware, safety and timing tests |
| 0E: Data and execution reliability | Sol 6.1 Extra High | Transition/context store, pending labels, eviction transactions, checkpoints, resume, quotas, scheduling, and CLI |
| 0F: Reporting and visualization | Sol 6.1 Extra High | Artifact-only summaries, progress, notebook, synchronized video, and statistical-analysis implementation against fixed definitions |
| 0G: Integration and acceptance | Sol 6.1 Extra High; bounded Astra numerical review | End-to-end smoke/recovery tests, real-model resource admission, numerical-boundary audit, and execution handoff |

#### Phase 0A: Astra Numerical Contracts

Before consumers implement their numerical calls, specify tensor shapes,
coordinate ordering, batch reduction, dtype, RNG ownership, and all fixed
conditioning inputs. Cover the seven-logit action representation, categorical
return bins, baseline/weight freezing, model-sampled detached score labels,
LTM/STM masks, FD probe scaling, and numerical versus statistical ridge.

Define the pure numerical interfaces separately from simulator, storage, and
scheduler effects. For example, return construction consumes recorded reward,
discount, and termination arrays; it does not inspect the simulator. The
EWC kernel consumes accepted parameters and a declared eviction batch; it
does not decide whether that batch has already been committed. The caller
must satisfy a documented transaction order and eligibility contract.

Provide tiny deterministic fixtures for NLL/KL, return telescoping, Fisher
diagonal, FD gradients, EWC admission/no-admission, and the definitions of
budget-averaged failure and $f,c,S$. These establish expected behavior without
requiring Sol to rederive the theory. Keep this contract concise and versioned;
do not create a second prose copy of this entire plan.

**Exit:** stable callable contracts and fixtures sufficient for independent
implementation. Reuse the same Astra numerical agent for 0B where practical.

#### Phase 0B: Astra Numerical Implementation

Implement and test the math-bearing slice: normalized joint likelihood and
its constants, exact directed/symmetric KL, model-induced Fisher diagonal,
per-record score sampling, FD perturbations and gradients, age weighting,
ridge/EWC scaling, eviction EMA, and the weighted actor/critic objective.
Include discount-product returns, return binning, frozen advantage weights,
and teacher logit conversion where they determine the likelihood inputs.

Use tiny models and float64 references to establish identities, then exercise
the intended accumulation dtypes. Specify how an effective model is evaluated
at perturbed adapter coordinates without detaching trainable dependencies.
The production Qwen loader, queues, archive implementation, notebook layout,
and Pi service belong to Sol, not this assignment.

**Exit:** tested numerical functions and a small objective/gradient reference
that Sol can run through the production adapter wrapper. Numerical acceptance
must include failure cases, not merely agreement at one benign parameter value.

#### Phase 0C: Sol Model And Language Plumbing

Implement the actual Qwen loading/quantization, explicit top-$L$ selection,
parallel adapters, role isolation, LSTM and conditioning path, recurrent-state
reconstruction, and cache invalidation. Wire one-time distillation and online
objective calls to the accepted numerical interfaces. Use separate source
files and tests from the Astra kernels; do not recreate their formulas inside
the training loop.

Test trainable-parameter inventories, adapter effective weights, language
conditioning, completed-instruction handling, role-specific gradients, and
the frozen-prefix backward boundary. Compare objective values and gradients
through the production wrapper against Astra's tiny-model fixtures. Unexpected
gradient routing is a numerical-boundary defect to resolve explicitly, not
an excuse to detach a costly branch.

#### Phase 0D: Sol Simulator And Robot Interface

Implement the backend-neutral protocol, primitive mapping, unit calibration,
simulator, continuous drift, camera/rendering, terminal-event logic, and
lightweight Pi service. Add watchdog, deduplication, authentication, timing,
and fake-driver tests. This work can begin alongside 0A/0B because it need not
wait for the numerical kernels.

Expose recorded reward components and elapsed-time inputs; call the numerical
return function instead of independently redefining discounting or episode
termination. Keep privileged evaluation state out of learner observations.
No physical robot connection or motor actuation is part of acceptance.

#### Phase 0E: Sol Data And Execution Reliability

Implement compact storage, replay/context eligibility, pending-return queues,
the exactly-once eviction ledger, durable simulator recovery, atomic model
publication, resource accounting, external-drive handling, and standalone
resume/status/report/stop commands. Call the accepted EWC kernel inside the
transaction contract; do not change its estimator to simplify persistence.

Test forced interruptions around every scientific commit boundary, including
the distinction between a repeated computation and a repeated admission of
evidence. Integrate control-priority GPU scheduling and bounded work queues.
This remains substantive engineering requiring tests, even though it does
not need Astra to author it.

#### Phase 0F: Sol Reporting And Visualization

Implement the concise notebook, live progress summaries, artifact readers,
held-out metric aggregation, paired intervals, sample-crossing/censoring
analysis, and synchronized egocentric/top-down video. Use fixed schemas and
synthetic fixtures while real trajectories are unavailable.

Follow the metric definitions in this plan and 0A fixtures. Do not introduce
different denominators, independent-frame uncertainty, or silent failed-crossing
exclusions. Keep the numerical estimation rules distinct from presentation
code so a targeted review does not require re-reading plotting machinery.

#### Phase 0G: Sol Integration And Focused Astra Review

Sol integrates the components, runs the acceptance suite, fixes routine
integration defects, and produces the handoff. Astra performs one bounded
review of the math-bearing integration: real adapter-coordinate gradients,
frozen likelihood inputs, FD normalization, Fisher score sampling, and the
eviction/update order. Include a compact check of the $f,c,S$ aggregation
against reference cases. Give Astra the relevant diff, fixtures, and test
results, not a request to reimplement the whole application.

Numerical defects return to the numerical owner with a reproducible failing
case. Sol owns repair of protocol, storage, CLI, renderer, and ordinary model
wiring defects. The numerical gate must pass before scientific runs, but
there is no requirement for Astra to supervise simulator profiling or each
later experimental batch. Sol retains the autonomous repair authority stated
below; material scientific changes still require versioning and disclosure.

#### Delegation And Token Discipline

Use bounded assignments with disjoint write sets, relevant plan sections,
specific acceptance commands, and an explicit output contract. By default,
start workers with a scoped brief rather than copying the entire conversation;
pass broader context only when it is genuinely needed. Each worker should
return changed paths, test results, unresolved limitations, and a short
integration note, not a second narrative plan.

Keep at most one Astra numerical worker active by default. Keep the number of
Sol workers modest, initially two, and reuse them for related sequential
assignments. Work packages are ownership units, not a requirement to spawn
seven agents. After 0A, model and data consumers can use the shared contracts
while 0B and the independent simulator work proceed. Reporting can start from
sealed artifact schemas and synthetic examples. Reserve shared-schema edits
for the coordinator to avoid conflicting patches.

Do not ask every worker to review the whole repository, duplicate a worker's
implementation in the parent, or run two independent full acceptance suites
without a concrete reason. The coordinator should do useful non-overlapping
work while delegates run, integrate each finished patch once, and close idle
workers. Serialize resource-heavy GPU checks; CPU coding/test concurrency
must also respect the node's current availability.

Record requested model/effort, task ownership, retries, and reported token or
credit usage where the harness exposes it. Mark unavailable usage as unknown;
do not infer exact billing from output length. This split targets lower
expensive-model usage, not a guaranteed reduction in total tokens: extra
handoffs and Extra High reasoning still have costs. Preserve the explicitly
requested reasoning levels unless the user changes them. No numerical
accuracy or recovery test is dropped merely to save tokens.

#### Common Acceptance And Handoff

Required fast tests, implemented by the corresponding owners above, include:

- Joint likelihood normalization and parameter-independent terms; fixed versus trainable weight misuse.
- Exact KL against a small reference; symmetric FD convergence and gradients through both endpoints.
- Joint Fisher formula versus per-record model-score sampling; detached synthetic labels and $g=-s$ signs.
- A toy weighted-mean update and its large-temperature Gaussian policy-gradient expansion.
- Adapter placement, parameter counts, role isolation, quantized forward checks, and LSTM effective matrices.
- Language reaches actions; completed commands are not reissued as new instructions; no future/privileged leakage.
- Action mapping, physical units, duration, watchdog, expired/repeated commands, and reconnect disarming.
- FOV, feasible resets, continuous gain/texture drift, terminal precedence, and potential telescoping.
- Delayed labels, episode interruption, retirement exactly once, and no archive admission on replay/sleep alone.
- Recurrent context eligibility, stale-cache invalidation, and pending-store bounds.
- Forced interruption/resume at action, label, update, retirement, and publication boundaries.
- Storage quota and external-drive loss; atomic completion; artifact-only reporting.

Provide a tiny CPU smoke configuration with a small deterministic encoder and
fake teacher/evaluator. It tests the pipeline, not Qwen capability. Add a
bounded CUDA smoke using the actual model only after resource admission;
model download and its storage/time are recorded, never hidden in a notebook.
Tests must never connect to a physical motor service by default.

Supply a standalone stage runner, status/report/stop commands, a pinned
environment manifest, and `PLAN19_HANDOFF.md` with exact next commands,
completed checks, open limitations, and budget counters. Validate in the
user's existing virtual environment; do not silently upgrade it underneath
other running studies. If dependencies conflict, propose an isolated robotics
environment and record its relationship to the main environment.

**Exit:** CPU acceptance, tested checkpoint recovery, working report skeleton,
and a reproducible profiling entry point. Do not declare simulator research
success because the smoke run produces a positive reward.

#### Phase 0 Execution Record

The immutable acceptance bundle is
`cache/numerical_experiment/picar/phase0/acceptance-v3` (280 tests passed;
no failures, errors, or skips). A fixed-seed synchronized scene preview was
also inspected. The notebook is pre-rendered from scalar artifacts and
does not require a model or training to open.

The initial CUDA diagnostic exposed FD step-size sensitivity. Before any
scientific runs, the adapted top transformer suffix and final norm were moved
to FP32 computation over still-quantized frozen weights, and small-output KL
arithmetic was stabilized. The frozen lower prefix remains BF16. A corrected
half/base/double probe gave penalty ratio 1.0472 and gradient cosines above
.9999, with gradients reaching both LTM and STM. Earlier diagnostics remain
available rather than being overwritten. The final acceptance bundle repeated
the corrected check against its fixed source snapshot.

The small probe reserved about 6.7 GiB on the 12 GB GPU. It used one image and
one history step, not the candidate context-8/batch-4 workload. Those larger
resource and real-time control settings remain uncalibrated. The standalone
smoke runner rejects production configuration; it is not a route for silently
substituting accelerated clocks or fake teacher/evaluator roles into science.

Other acceptance repairs covered independent paired optimizer/RNG state,
zero-age indexing, exact observation-ID recovery, and immutable nested-run
boundaries. A separate clocked physics worker and bounded control session keep
acquisition progressing under delayed inference/updates. Fallback intervals
are tagged, not invented as Gaussian policy labels. See the handoff for the
remaining stage-integration and physical-readiness boundaries.

### Phase 1: Sol Physical Readiness And Resource Calibration

**Purpose:** establish that this is a controllable task and a feasible online
system before spending the remaining budget on learning curves.

Run the scripted feasibility controller, camera/mapping audit, frozen reward
sanity panel, Qwen latency probe, and end-to-end control/update profiler.
Measure actual GPU free memory; reserve headroom rather than filling all VRAM.
Use available physical measurements if supplied. Missing physical calibration
does not block simulated learning, but it blocks strong sim-to-real claims.

Stage the profile rather than crossing every knob:

1. Fix a usable sensor/token/context configuration and a safe control period.
2. Test top-$L$ choices 1, 2, then 4 only if headroom permits.
3. Test LSTM widths 64/128, LTM ranks 4/8, and STM rank 1; rank 2 is optional if clearly feasible.
4. Measure online batches 1/4/8 and context lengths 8/16 at the shortlisted model.
5. Compare one/two/four eviction score draws and one/two FD probes on a bounded batch.

Prefer the smallest model that preserves task competence and language
conditioning. A top-zero-block frozen-feature fallback may establish a
diagnostic baseline, but cannot silently replace the agreed top-$L$ experiment.
If recurrent LoRA is numerically problematic, fix and test it or propose a
clearly versioned placement change for both conditions.

Publish VRAM, forward/backward/FD/Fisher times, teacher/reward throughput,
deadline misses, bytes per transition, projected checkpoint cost, and the
20-hour resource admission forecast. Include evaluation and sleep in the
forecast. Inspect egocentric/top-down clips before accepting simulator geometry.

**Exit:** common feasible model and timing recipe, documented uncalibrated
physical assumptions, and measured cost ranges. If no feasible configuration
exists, preserve diagnostics and request a concrete scope/resource decision;
do not silently turn the experiment into offline imitation.

#### Phase 1 Execution Record (2026-11-09)

The diagnostic campaign `cache/numerical_experiment/picar/phase1/resource-v2`
finished and stopped for the requested resource check-in. Nine staged CUDA
probes, the scene audit, frozen-role checks, and an independent-clock pilot
are sealed as separate immutable units. The main campaign took 6.1 minutes
of active execution; earlier diagnostics, implementation, and tests are
additional. No Phase 2--5 learning or physical robot motion occurred.
**The profiling work is finished, but task/online-readiness exit gates have
not passed.**

The memory shortlist is top-$L=1$, LSTM width 64, LTM/STM ranks 4/1,
history 8, replay batch 1, one FD probe and one eviction score draw. Inputs
are 160x120 RGB with a 19,200-pixel processor limit. It reserved 7.11 GiB
on the 12 GB card, leaving 3.85 GiB after initial non-PyTorch use. Top-$L=2$,
batch 4, and context 16 ran out of memory; top-4 and batch-8 were consequently
skipped. Width 128, rank 8, two FD probes and two/four Fisher draws were
measured successfully, but did not justify enlarging the resource shortlist.
This is not yet a competence-selected model.

| Operation, selected shortlist | Median seconds |
|---|---:|
| Cold action inference, history 8 | 0.790 |
| Control fit: one new plus one replay transition | 1.714 |
| FD fit: same new/replay allocation | 1.980 |
| One retired-transition Fisher, history 8 | 0.866 |
| Frozen teacher / evaluator | 0.172 / 0.169 |
| Durable checkpoint transaction, 8.22 MB learner state | 0.056 |

Fits include their own NLL/FD work; do not add the isolated backward probes
to these fit timings. Every event begins with a cold encoder cache and the
new/replay inputs differ. An exact, bounded CPU cache reuses only frozen
prefix activations within an event: adapted suffixes and FD endpoints remain
differentiated. Frozen roles now encode only the final independent RGB row,
whose recorded text/action context is unchanged; previously computed earlier
role rows were discarded. CPU output/gradient equivalence tests cover both
optimizations. The initial uncached diagnostic remains in `resource-v1`.
The smoke FD coefficient is not a selected scientific penalty strength.

| Readiness check | Finding |
|---|---|
| Privileged scripted controller, actual calibrated action mixtures | 10/10 A/B/mirrored layouts succeeded; valid stopped hold |
| Camera/control mapping | Passed; synchronized ego/top-down clips inspected |
| Frozen Qwen teacher | 0/2 A and 0/2 B diagnostic rollouts succeeded; all collided |
| Frozen completion potential | Did not order success above far failure; collision panels scored highest |
| Independent-clock pilot | Physics continued through two ~2.1 s updates, but 6/17 non-startup control intervals used safe fallback at a 1.25 s period |
| Physical PiCar measurements | Not supplied; no hardware validation claimed |

Four teacher rollouts are a readiness warning, not a precise success-rate
estimate or proof that Qwen cannot learn/control this task. Likewise, the
two-update pilot does not admit continuous one-update-per-acquisition
operation. Resolve teacher/reward extraction and the action/update scheduling
envelope before collecting scientific bootstrap data. A safe fallback is
tagged as such, never relabeled as a valid sampled Gaussian policy action.

| Conditional whole-study scope | Bootstrap | Development | Fresh pairs | Closeout | Total, including recorded early work |
|---|---:|---:|---:|---:|---:|
| Nominal: 6 pairs, 2,400 new transitions/arm | 0.6--0.9 h | 5.8--9.3 h | 32.1--44.5 h | 0.25--0.75 h | **38.9--55.7 h** |
| Smaller permitted size: 4 pairs, 1,200/arm | 0.6--0.9 h | 5.8--9.3 h | 12.1--16.8 h | 0.25--0.75 h | **19.0--28.0 h** |

These are engineering allowances, not confidence intervals or authorization
to exceed 20 hours. They include shared 600-transition bootstrap, 8--12 short
development trajectories, seven held-out checkpoints/72 evaluation episodes
per arm, sleep and checkpoint I/O. They assume one optimizer event and one
eviction Fisher per new transition; larger sleep batches, teacher rescue and
additional solver steps need remeasurement. Full-history Fisher timing is
conservative when FIFO eviction masks already-retired context. Physics and
GPU time overlap rather than being counted twice. No affordable sound
whole-study configuration is yet guaranteed.

Storage is another admission issue: raw training frames/records project to
13.5 GB nominal or 4.7 GB smaller, but retaining every full transaction
snapshot projects to **1.55 TB / 393 GB** respectively, before development,
videos and failed attempts. The nominal policy exceeds the proposed 1 TB
archive quota; either exceeds the 100 GB local quota. Before learning, use a
bounded recovery-checkpoint policy with append-only event records and an
explicit archive root. Do not silently consume the external drive or prune
completed artifacts. The Phase 1 artifacts themselves occupy about 95 MB.

#### Phase 1 Fix 1: Grounded Roles And Feasible Online Execution

**Motivation:** the initial resource campaign exposed readiness defects, not
an STM learning result. Scripted control succeeded on 10/10 layouts, but Qwen
teacher rollouts all collided, the completion potential ranked collisions
above success, and whole-update model exclusion caused 6/17 non-startup
intervals to fall back. Repair the interface and execution envelope before
spending the scientific learning budget. Preserve every original artifact.

**Scope and autonomy:** Sol implements and runs a separately versioned,
resumable readiness pass under `cache/numerical_experiment/picar/phase1/`.
No Phase 2--5 learning or physical actuation is authorized by this repair.
Aim for at most three additional active compute hours, including failed
attempts, with engineering time reported separately. Start with small stored
development panels, permit two bounded role rescue attempts, and stop with
clear diagnostics if useful task competence is still absent. Do not turn
weak teacher behavior into an apparent negative STM result. Update the
concise results notebook after each completed diagnostic unit.

1. **Audit the model interface before tuning it.** Compare native Qwen
   forward/generation with the split encoder on identical RGB/text inputs,
   accounting explicitly for the FP32 adapted suffix. Check selected-token
   logits, precision, preprocessing, positional masks, and batch isolation.
   Record unconstrained descriptions, total probability on permitted answers,
   and mirrored-image/contrasting-instruction responses. Restricted digit
   softmax is an unvalidated decision interface, not calibrated uncertainty.
   Native generation and multi-image input are supported reference routes,
   not assumptions about task competence. [Qwen interface][qwen]
2. **Ground the teacher and visual potential.** State each primitive's real
   drive/steering/camera effect and the units/order of recorded controls.
   Explicitly name the neutral command's stopping effect rather than only
   calling it "look-forward." Compare the retained legacy interface against
   clearer constrained answers, a short timestamped RGB sequence and a
   modestly higher frozen-role resolution. Keep privileged state, panel IDs,
   desired labels and future frames out of model inputs. Preserve the
   probability-weighted control map. If hard generated choices are softened
   for execution/distillation, apply the probability floor before execution
   and label them `hard_engineered`, never calibrated soft probabilities.
   Describe visible completion/progress explicitly and test its ranking on
   separate layouts; monocular images cannot certify exact physical distance.
   Select using development only, then audit a fresh A readiness panel of
   20 layouts and report B behavior separately. Existing 70% teacher guidance
   remains a readiness guide, not a significance claim. No teacher or reward
   fitting may consume the future scientific confirmation panel.
3. **Make scheduling genuinely inference-priority.** Drop expired work before
   expensive inference. Permit cooperative yields only at graph-safe boundaries
   where no functional parameter override remains installed and actor-visible
   parameters have not changed. Serialize GPU work; do not claim thread-level
   overlap removes its cost. Inference may read the last published parameters
   while gradients are being accumulated, but publication/optimizer mutation
   remains atomic, with exactly one publication per accepted event. Preserve
   actor exploration RNG consumption if a yielded update rolls back. Test
   gradient equivalence, failures, competing writers, expiry and recovery.
   Profile cold action latency, maximum uninterruptible work, fallback rate,
   update throughput and bounded deferred-label work. Recheck scripted
   controllability before selecting a slower common control period or shorter
   context. Do not pause physics, fabricate policy actions, or silently move
   all learning to episode boundaries to pass a deadline test.
4. **Bound persistence rather than repeatedly copying history.** Offer a
   separately versioned checkpoint/journal policy with current/previous
   recovery state, explicit milestones and compact append-only evidence and
   transaction records. Validate atomic publication, halt recovery, idempotent
   retirement/event handling and quota accounting before using it. No old
   completed artifact may be rewritten or pruned. Keep legacy storage readable
   and default behavior explicit. Archive roots remain arguments; no personal
   paths or external source-tree imports enter project code.

The simulator's terminal success/collision/deadline labels are distinct from
the VLM shaping potential. Because complete-return shaping telescopes as
derived above, a no-shaping variant is a legitimate **explicitly versioned
fallback for review**, not permission to silently replace the primary reward.
Likewise, a between-episode-only update variant must be proposed and labeled,
not passed off as successful between-action online execution. Implementing
diagnostic alternatives does not select them for the later learning study.

**Deliverables:** tested repair code and `--resume` entry point; immutable
native/custom audits, role panels/rollouts and real-time scheduling diagnostics;
before/after readiness summary and progress in `plan19_results.ipynb`; amended
resource/storage forecast only after a feasible common recipe is identified.
Keep failed attempts visible. Report unresolved gates without inventing a
competent teacher or an affordable full study. The mathematical likelihood,
diagonal eviction-only EWC, STM FD objective and
`mathematical_overview.ipynb` remain unchanged.

**Implementation refinements:** preserve the initial Fix 1 revision, including
its failed role tests. The follow-up revision keeps the original grounded
prompt available (`prompt_version=1`) and explicitly appends current active
instructions in version 2, without reactivating completed instructions or
repeating a competing default mission. An optional cadence-aware braking
rule retimes only the privileged feasibility controller: cap approach speed
by the distance to the stopping neighborhood divided by the control period.
It changes neither robot dynamics nor learner controls and supplies no
primary demonstrations. History-4 scheduling alternatives remain common-model
development candidates, not an automatically selected scientific condition.
Compare shorter periods with controllability before accepting long deadlines
that happen to eliminate missed inference. Both repair revisions count toward
the same three-hour compute allowance.

**Fix 1 execution record (2026-11-09):** implementation and bounded diagnostics
are complete in immutable `phase1/fix1-v1` and `phase1/fix1-v2` campaigns.
The two campaigns consumed 9.8 recorded active compute minutes, excluding
implementation/unit-test time. Neither starts Phase 2--5. The newer campaign
contains a checksummed source bundle and operational recovery notes; original
failures remain archived and are not pooled with its revised role interface.

| Repair check | Observed result |
|---|---|
| Native versus split Qwen, matched precision | Exact hidden-state agreement on all four diagnostic scenes |
| Restricted answer validity | About 99.8% vocabulary mass on the permitted digits; rare-token normalization was not the failure |
| Live language wiring, revised interface | Stop, left and right contrasts pass for both revised restricted and generated interfaces; original grounded prompt remains available |
| Context-4 cooperative timing, periods 0.75 / 1.0 / 1.25 s | 0/20, 0/15 and 0/13 non-startup fallbacks, respectively; two diagnostic optimizer events per pilot |
| Context-4 atomic comparison, same periods | 4/24, 2/17 and 0/12 fallbacks; denominators differ because these are closed-loop timing pilots, not a paired task-effect estimate |
| Cadence-aware scripted feasibility | 10/10 layouts at 0.75 s and 1.0 s; 8/10 at 1.25 s |
| Revised teachers, closed-loop development | Both interfaces still failed their two A and two B rollouts; 20-layout fresh validation was therefore not attempted |
| Visual completion potential | Still failed the success-versus-far/collision ordering check |
| Storage and recovery | Opt-in `journal-v1` passes crash/idempotency/legacy compatibility tests and is used by repair runs; existing completed data is untouched |

Context 4 with a 1.0 s control period is a promising **burst-timing candidate**,
not a selected complete learning recipe. Two optimizer events do not prove
sustained one-update-per-observation throughput, deferred-label capacity, or
student competence. No favorable runtime revision is claimed until those
questions and teacher readiness are settled. Current blockers are teacher
navigation skill and visual reward quality, not native/split forward parity
or the basic verbal instruction connection. A different bootstrap/control
strategy or explicit reward fallback requires the subsequent scope discussion.

A Tk GUI backend abort interrupted the second pass. Headless Agg rendering
was enabled, completed units were verified/skipped with `--resume`, and the
incomplete short timing pilot was retried. No scientific settings changed for
that recovery. This operational event is also recorded in the notebook and
the source-bundle artifact.

#### Phase 1 Fix 2: Restore Dense Visual Rewards And The Robot Workflow

**Status and motivation (2026-11-09):** drafted by Astra for review; Sol
implements and executes after authorization. This is a separate amendment
following the completed **Phase 1 Fix 1**, not a replacement of that work.
The initial, unexecuted draft was nested under Fix 1; the user corrected its
name to Fix 2 and requested explicit inclusion of the related corrections.
Keep Fix 1's specification, execution record and artifacts intact. Future
material revisions should append a numbered amendment and an execution
record, with only status/index pointers maintained in earlier sections.
The user reports
successful frozen, quantized Qwen guidance and graded red-ball rewards on the
physical robot. Inspection found three departures in this implementation:
binary completion-token scoring replaced explicit numeric visual progress;
potential differences replaced accumulated visual reward; and readiness
required obstacle avoidance plus precise stopping from the outset. The
failed diagnostics therefore do not establish that Qwen cannot provide the
intended teacher or reward. Reproduce the simpler behavior before making that
claim or studying STM efficacy.

The earlier project's `src/picar_kl/reward/prompts.py` and `reward/scoring.py`
provide the implementation reference: generated `{"reward": number}` in
[0,10], with zero for an absent ball and graded values as it approaches.
The checked-in prompt peaks at a large, fully visible ball and penalizes an
overfilled image; it does **not** explicitly reward contact. The user's live
experience may have used another configuration. Record this distinction;
describe the contact task below as a proposed revision matching the user's
stated intent, not an exact recovery of an identified old run. Copy necessary
code/prompts with provenance into this repository; another checkout is never
a runtime dependency or required installation asset.

**Scope:** repair and verify the common acquisition/reward system, implement
the missing default-off retired-sleep option and physical/speech interfaces,
then stop for the resource check-in. Physical interface acceptance uses
injected drivers and recorded inputs. Do not launch Phases 2--5, enable retired
replay in the primary experiment, or operate the physical robot.
Retain the shared quantized backbone, top-$L$ adaptation, additive STM/LTM,
eight-action weighted mapping, one-time distillation, normalized joint NLL,
diagonal eviction-only EWC and symmetric STM FD-KL. New task/reward semantics
receive new identities; historical artifacts remain readable and immutable.
The following instructions supersede the earlier Fix 1 binary-potential
readiness criterion and task/reward defaults for the new revision only.

| Correction | Fix 2 implementation and acceptance |
|---|---|
| Graded visual rewards and return resolution | Items 1 and 3: generated scores, accumulated rewards, bounds, bins and weight diagnostics |
| Target contact versus unsafe collision | Item 2: explicit task version, event precedence and separate reported outcomes |
| Instructions, corrections and completed-action history | Items 2 and 6: causal conversational context, timestamped text/speech events and replay/resume tests |
| Public time remaining | Item 2: explicit countdown in actor/teacher inputs, context identity and reconstruction |
| Retired-data replay during sleep | Item 5: functioning default-off option with an explicit evidence-accounting rule |
| Real Pi actuator/camera and live speech interfaces | Item 6: project-local implementations, declared optional dependencies and no-hardware acceptance tests |

All six are deliverables, not observations left in an audit table. Actual
physical calibration/operation and scientific use of retired replay remain
separate follow-on work. The implementation status below is not a claim that
these features already exist.

**1. Reproduce and diagnose the role interfaces.**

Port the earlier system/user message separation, numeric JSON reward prompt,
and named-action JSON generation before adding new abstractions. Preserve
raw generated text, parse outcome, prompt hash, preprocessing, input tokens,
model revision and latency. Reject nonfinite/out-of-range or malformed
scores; bounded retries are recorded and never fabricate reward zero.
Compare the exact old score rubric with one clearly versioned task-aligned
rubric. Ordinary generated action choices remain `hard_engineered` after
the declared probability floor, applied before execution/distillation.
Keep the probability-weighted control vector, including camera channels.

Use existing RGB artifacts first, then a compact development panel with
four layouts and six visible situations per layout: absent target, far,
intermediate, close, partly occluded, and excessively cropped/too close.
Include mirrored views and A/B textures. Construct a disjoint four-layout
validation panel before selecting the interface. Privileged geometry may
construct/audit these panels but cannot enter model prompts, file captions,
features or reward values. Do not give Qwen the intended order or state name.
The exact legacy rubric may decrease at excessive proximity; test that
separately from the valid approach interval.

Show a small contact sheet with raw scores and an approach-distance versus
score plot. Require finite valid output, a useful absent/present separation,
and mostly correct far-to-near ordering within matched views. Use 80% correct
strict order on clearly separated approach pairs as an engineering guide;
report ties and the denominator, and inspect failures rather than treating
this small panel as a significance test. Do not require every success image
to outscore every collision image: visible proximity and physical safety are
different components. A car close to an obstacle may still see a close ball.
The separate terminal penalty handles that event. Compare the new score with
the archived completion probability without treating their scales as equal.

**2. Give navigation a diagnostic ladder and a coherent task definition.**

The proposed new full task, `ball-contact-v1`, keeps the existing arena,
obstacle, camera/controls and continuous texture/steering drift. Target
contact ends the episode successfully and stops the drive; wall or obstacle
contact is failure and takes precedence if simultaneous. Floor/wheel support
contact is not a terminal event. Retain the 20-second deadline and +10/-5/-1
success/collision/deadline bonuses. Do not impose the old clearance band,
heading tolerance or one-second stopped hold on target-contact success.
Retain `safe-stop-v1` and its tests for old runs and optional diagnostic
comparison. Report target contact and unsafe contact as separate events.
The future real backend needs an explicitly validated contact observation
(sensor or operator annotation); RGB alone is not assumed to certify it.
This remains a hybrid simulator reward, with Qwen supplying dense visual
scores and simulator contacts supplying terminal events.

Run at most six development layouts at each of two stages: unobstructed
ball approach, then the full obstacle task. Keep tasks/denominators separate.
Test the direct port first; allow one targeted correction informed by the
failure clips, such as clearer action semantics, history or image resolution.
Avoid an open-ended prompt search. Passing the unobstructed stage alone
does not admit the full task. If it alone works, finish with an explicit
recommendation to simplify the later experiment; do not silently collect
primary bootstrap data on an easier task.

Use the earlier conversational context pattern as well as the visual input.
The current `phase1_roles._batch` always supplies one fixed instruction:
replace that assumption in the new runner with an explicit, timestamped
instruction/history argument. Test a new instruction, a correction after
executed actions, and a completed instruction that must not restart.
Retain short prior robot status messages as fallible history, not new ground
truth. Include previous available visual scores if the ported interface uses
them, respecting their actual availability timestamps. No future reward or
terminal result may appear in pre-action context. A visible-history-based
coaching policy is allowed, fixed in development and reported separately from
unassisted teacher performance. It may not read simulator pose or an oracle
controller. Human coaching remains optional and explicitly tagged.

Expose episode time remaining explicitly to the actor/teacher context:
`models.render_context` currently supplies timestamps but no deadline field.
Version the context schema and include this known task clock in cache keys,
replay reconstruction and resume tests. It is public task information, not
future outcome leakage. Preserve the finite-horizon likelihood assumptions.

If the full task becomes usable, freeze the selected interface/coaching
recipe and run the existing 20-layout fresh A readiness check; report B
diagnostics separately. Preserve the 70% teacher guide and every failure.
Coaching-assisted success must be named as such, and no fresh-panel feedback
may tune that recipe. Inspect synchronized ego/top-down clips at both stages.

**3. Restore reward density through the entire return pipeline.**

Let the frozen, RGB-only evaluator generate $v(h)\in[0,10]$. For the new
`dense-visual-v1` reward, propose

$$
r_t=\beta\frac{\Delta t_t}{T_{\max}}\frac{v(h_{t+1})}{10}
     -c\frac{\Delta t_t}{\Delta t_0}+b_t,
\qquad
\gamma_t=\gamma_0^{\Delta t_t/\Delta t_0},
$$

with $T_{\max}=20$ s, $\Delta t_0=0.5$ s, $\gamma_0=.99$,
$\beta=2$, $c=.02$, and the terminal bonus $b_t$ specified above. This
normalizes the visual reward into at most two undiscounted units over a full
episode, leaving successful completion substantially more valuable. It is
an explicit scale choice for the new task, not a claim that the earlier robot
used these weights. Actual elapsed time, including a partial final interval,
sets the exposure weight. The undiscounted visual reward budget depends on
elapsed time, not picture count. Keep control cadence common across methods;
this weighting does not assert identical trajectories or exactly identical
discretized discounted returns after changing cadence. Do not multiply
terminal bonuses by elapsed time or zero the visual score solely because an
episode terminates.

Keep cached evaluator outputs frozen per RGB/history/prompt/model identity.
Use recorded post-action observations only, without future frames. Deferred
scoring/labeling does not invent new acquisitions. The visual score is now a
stage reward, not $\gamma_t\Phi_{t+1}-\Phi_t$. Thus it remains in the complete
return instead of telescoping to a starting-state offset. Potential-based
policy invariance does not apply to this deliberate objective revision.
Both learning conditions must use the same revised reward. [Original
potential-based shaping rationale][shaping]

Positive progress rewards can encourage lingering. Before admission, test
scripted reward sequences for immediate success, delayed success while
maximizing visual score, visible-ball loitering until deadline, disappearance,
and collision. With the proposed scales, prolonged maximum visibility must
not beat timely success. These are reward-arithmetic audits, not primary
demonstrations. Report actual teacher behavior as well: a bounded reward
does not guarantee that the learned policy optimizes it well. If a scale
correction is necessary, freeze it from these common development diagnostics,
record the reason, recompute bounds, and do not tune it separately by method.

Derive bounds before any new initial likelihood fit. For these defaults,
conservative bounds are

$$
-5-cT_{\max}/\Delta t_0\leq G_t\leq10+\beta,
\quad\text{so use an enclosing range }[-6,12].
$$

The existing `LearningConfig` range [-8,11] must not be inherited by accident.
Keep out-of-range returns as errors, never silent clipping. Start the new
recipe with 41 bins and inspect 81, then at most 161 only if needed; this is
a bounded resolution check, not a treatment factorial. On recorded or
constructed paired sequences with the same terminal outcome/time and visibly
different approach histories, report raw return differences, bin ties and
the resulting advantage-weight differences. Ensure useful graded differences
survive discretization before declaring reward density restored. Also report
the log-weight error bound (half the maximum bin width divided by $\tau$),
weight concentration and any cap hits. If no affordable resolution preserves
the signal, report the issue rather than change the likelihood family.

The categorical-return/Gaussian-action likelihood, score definition, Fisher
and exact joint KL formulas remain valid with this new fixed label recipe.
However, bins, baseline scale, temperature and critic initialization must
match the new reward. Version all of them together; rebuild the shared
bootstrap/initial Fisher when Phase 2 is later authorized. No old return
labels, critic states or EWC summaries are silently reused across versions.
The raw observations may be explicitly re-scored into separate derived
artifacts with parent checksums, never edited in place.

**4. Implement narrowly, verify the chain, and reforecast.**

| Component | Sol implementation instruction |
|---|---|
| `role_repair.py`, `teacher.py` | New generated numeric scorer and versioned conversational teacher; preserve old role implementations |
| `src/robotics/geometry.py`, `simulator.py` | Explicit task version; distinguish target versus unsafe contact, terminal precedence and exact resume |
| `models.py`, role history builders | Actual causal instruction/status history and public time remaining; cache/replay identity includes both |
| `src/continual/picar_math.py`, `NUMERICAL_CONTRACT.md` | Add a separate dense-reward helper and documented contract; retain `shaped_rewards` and its historical telescoping tests |
| `config.py`, return-label orchestration, `learner.py` | Reward/task versions, derived bounds/bins and scale provenance; audit label-to-weight-to-gradient behavior |
| `phase1_fix*.py`, schema/reporting | New resume-safe campaign identity and explicit raw score, stage reward, terminal reward and return fields |

Add focused tests for strict numeric parsing, frozen evaluator isolation,
causal language/deadline handling, contact types, reward bounds, partial
intervals, no terminal-bonus duplication, bin/weight resolution, and exact
recovery before/after label commits. Keep the normalized-NLL, KL, both-bank
FD gradient and eviction-only tests passing. Compare teacher/evaluator native
and custom routes only where changed preprocessing or generation warrants
it; do not rerun every completed resource sweep. Never modify
`mathematical_overview.ipynb`.

Profile generation cost again: the earlier ~0.17-second restricted-token
role timings do not forecast full JSON generation. Run a short multi-episode
independent-clock diagnostic that includes actual frozen reward calls,
completed-return labeling, bounded learner updates and eviction work for
both loss settings. It is a pipeline/timing test, not a learning comparison.
Use the cooperative scheduler and measure queue growth/drain, fallback
fraction, maximum uninterruptible work, update batch size and publication
age. Do not claim sustained service from another two-update burst or hide
deferred scoring/retirement after the timed window. If work exceeds the
online window, use the declared bounded update/sleep schedule and report its
actual service rate; do not silently freeze physics or label offline updates
as between-action learning.

Use new immutable units (suggested family `phase1/fix2-v1`), explicit
`--resume`, journal checkpoints and a source snapshot. Store task, rubric,
reward and coaching versions in the identity and reject incompatible resume.
Reuse original RGB data read-only; use new layout seeds for validation.
Refresh the notebook from artifacts as units finish, with a short explanation
of why prior results did not test the intended dense reward. Show only a
contact sheet, score plot, staged success table, timing/backlog summary and
concise conclusion; link raw traces instead of adding a wall of numbers.
Do not edit old numerical outputs or present planned corrections as results.

**5. Complete the optional retired-data sleep path.**

Implement `retired_archive_replay` as a working, explicitly versioned,
default-off capability instead of an unconditional configuration rejection.
The main Replay/EWC-versus-STM comparison continues to use eligible replay
only, and its resolved configuration must reject accidental activation.
This repair validates the option with tiny CPU fixtures, not an additional
scientific treatment or performance rescue.

Use a declared `supplemental-rehearsal-v1` protocol: during a stopped sleep
interval, replay a bounded sample of archived, fully labeled transitions
with their original frozen likelihood designs and reconstructible histories.
Reject incompatible task, reward, model or label-design versions instead of
mixing archived observations from a different experiment.
Combine eligible and archived items in one replay batch mean, with a fixed
configured archived fraction and the same total batch/step budget. Defaults
for an explicitly enabled test are archived fraction 0.25 and age 1 for
archived items' STM penalty; neither is selected for scientific efficacy.
Report the realized source counts when a source has insufficient eligible
records. Never replace missing archived data with fabricated observations
or silently expand replay eligibility during ordinary online updates.

These observations already contributed to EWC. Retain the existing EWC
term, explicitly describing the extra likelihood exposure as supplemental
rehearsal, not an independent new observation or an exact disjoint-data
likelihood decomposition. Do not attempt to subtract a sample from an
aggregate diagonal EMA without attribution information. Archived replays
must not increment new-data/sample counts, drift exposure, retirement counts
or EWC admissions, and must not recenter the archive or compute a second
retirement Fisher. A later scientific study of this option must account for
the deliberate repeated use of information; the default primary experiment
does not incur it.

Fetch archived context through an explicit permission/eligibility path.
Charge frame/context bytes, read/decode time and optimizer work; retain only
a bounded cache. Preserve checksums, labels, baselines and retirement IDs.
Unavailable archive storage pauses the requested operation with a recorded
reason. Persist source-sampling RNG, planned batch IDs and optimizer commit
receipts so resume neither resamples committed work nor repeats admissions.
Test default-off behavior, enabled selection, exact counts/EWC invariance,
archive failures, bounded context access and halt/resume around sleep commits.
Update CLI/configuration documentation and remove the blanket rejection only
once these tests pass.

**6. Complete the transferable robot and speech interfaces in software.**

Provide a project-local physical driver implementing the existing
`demo/picar/service.py` `Driver` contract (`apply`, `stop`, `capture_rgb`).
Port the minimal actuator/camera functionality from the earlier project,
recording provenance and any third-party license requirements. Its legacy
`robot/legacy_car_env/car.py` performs setup, camera opening and servo writes
at import: do not copy those side effects. Load hardware dependencies lazily
and perform initialization only in an explicit physical-service launch.
Keep fake-driver launch as the default for automated acceptance. Implement
hardware-specific SI-to-servo/speed conversion through explicit calibration
configuration; do not present guessed calibration constants as measurements.
Camera conversion, resolution, timestamps, failed reads and shutdown must
follow the same backend contract used by simulation.

Keep the Pi's dependencies lightweight and declared separately; Qwen, Torch,
learning and speech-model inference remain server-side. Document how a fresh
clone installs the optional hardware dependencies without another checkout.
Preserve service authentication, session/sequence deduplication, expiry,
watchdog stopping and explicit arming. Write injected wheel/servo/camera
tests covering signs, units, saturation, stop, capture failures, reconnect
and shutdown. Do not import real hardware modules or connect to the robot
during acceptance. Deliver an installation/calibration procedure for the
later operator session and label physical behavior as unvalidated until then.

Wire a common timestamped instruction-event interface from typed input and
optional server-side speech transcription into the same history used by
teacher, learner and replay. Reuse the earlier speech/stream design with
project-local code and declared optional dependencies. Support recorded
audio/transcripts for unattended tests; model assets download to this
project's cache unless an explicit reuse path is supplied. User microphone
input remains optional for completing simulated experiments. Speech inference
must not block capture, control deadlines or the watchdog; include its cost
if enabled in a later operational run.

Store event IDs, utterance timestamps and the time a transcript became
available. An utterance cannot influence actions before its text was available.
Corrected, superseded and completed instructions retain causal provenance;
resume must not inject the same speech event twice. Test typed/transcribed
equivalence, delayed transcription, interruption/correction, completion,
duplicate delivery and replay reconstruction. A transcript/status interface
may support robot replies, but ordinary student motor actions continue to
come directly from the LSTM; do not reintroduce per-action language generation
after the one-time distillation. No microphone, speaker or robot access is
needed for these acceptance tests.

**Deliverables and time accounting:** include all six work items in the
handoff and notebook progress summary. Mark software acceptance separately
from physical validation, and mark default-off feature support separately
from a scientifically executed treatment. Sol should finish the no-hardware
interface tests even if navigation remains weak, preserving useful engineering
results without claiming experiment readiness.

| Work | Additional active wall-clock estimate |
|---|---:|
| Interface reproduction, RGB score panels and reward arithmetic | 0.1--0.3 h |
| Staged teacher diagnostics and conditional fresh readiness | 0.3--0.8 h |
| Return resolution, real-time pipeline and recovery checks | 0.3--0.8 h |
| Retired-sleep and physical/speech interface acceptance, no hardware | 0.1--0.3 h |
| Notebook, artifact audit and revised study forecast | 0.1--0.2 h |
| **Total** | **0.9--2.4 h** |

Allow roughly 4--9 hours of Sol implementation/debugging separately, revised
from 2--5 hours to include items 5 and 6. These are planning estimates. Keep
Fix 2's charges separate from the completed Fix 1 ledger while counting both
against the existing three-hour repair envelope and overall study ledger.
Fix 1 consumed 9.8 recorded minutes; renaming the new work does not reset the
allowance or grant a fresh three hours per attempt. Show both phase totals
and cumulative charges on resume. Report a measured reforecast if needed.
Report the existing 39--56 / 19--28-hour study forecasts as superseded
conditional estimates until the revised path is measured. Finish with a
measured full-study proposal and remaining concerns, not automatic Phase 2.

**Alignment audit and assigned corrections:**

| User requirement | Current finding | Required disposition |
|---|---|---|
| Graded Qwen rewards for visible approach | Binary completion score plus telescoping shaping differed materially | Repair here, including return resolution and scales |
| Reproduce useful initial Qwen control | Fixed-instruction tests omitted the planned coaching/conversation path; full stopping task was harder than simple approach | Repair and report staged, assisted/unassisted competence separately |
| Contact as strong success evidence | Old task treats target contact as failure; checked-in old prompt itself does not specify contact | Proposed explicit `ball-contact-v1` revision above; preserve old task and label hybrid terminal supervision |
| Language-conditioned LSTM, one initial distillation | Architecture/stage plan is aligned; conversational rollout integration is incomplete | Finish causal history here; complete distillation/student readiness in Phase 2 |
| Small top-$L$ STM/LTM and eviction-only diagonal EWC | Implemented architecture/numerical path matches the intended design | Preserve; the readiness failures do not motivate changing STM mathematics |
| Online updates plus recharge/sleep work | Cooperative machinery exists, but sustained load remains unmeasured | Measure here; carry the admitted service/sleep schedule into later runners |
| Optional retired-data use during sleep | Config currently rejects it despite earlier prose claiming implementation | Fix 2 item 5 implements and CPU-tests the default-off supplemental-rehearsal protocol; primary scientific runs keep it off |
| Maintained real Pi deployment | Shared client/service and fake-driver tests exist; actual actuator/camera adapter and live speech bridge are not delivered | Fix 2 item 6 implements both interfaces and no-hardware tests; real calibration/actuation follows separately |
| Reproducibility, storage, reader-facing progress | Project-local model defaults and bounded journals are aligned | Preserve; use externally supplied archive paths and include all deferred work in forecasts |

No additional architecture redesign is indicated by this audit. The primary
repairs concern what the robot is asked to do, what is rewarded, how that
reward reaches the likelihood, and which planned interfaces are actually
connected in the acquisition loop.

**Fix 2 execution record (2026-11-09):** all six implementation work items
are delivered. The completed corrected campaign is
`cache/numerical_experiment/picar/phase1/fix2-v2`; its software acceptance
passed **637 tests**. The first attempt, `fix2-v1`, is retained with a stop
request and immutable completed units. Its parser rejected otherwise valid
JSON in a single Markdown fence. Version 2 accepts bare JSON or one exact
JSON fence, retaining strict keys, types and score ranges. Its timing replay
also masks retired recurrent predecessors. These changes have new source
identities and exact-source snapshots; neither attempt was overwritten.

| Deliverable | Executed result and remaining qualification |
|---|---|
| Dense visual rewards and contact semantics | Valid generated 0--10 scores; contact-aligned rubric correctly ordered 5/12 development and 8/12 held-out approach pairs, below the 80% guide. Absence/close separation passes, but far/intermediate ties remain common. Historical reward/task versions are unchanged. |
| Staged teacher navigation and causal context | Ported interface reached the ball in 3/6 unobstructed episodes; longer grounded interface achieved 0/6. Full obstacle task achieved 0/12 across both interfaces. Rollouts used accelerated physics for diagnosis, not real-time teacher admission. |
| Return likelihood and resolution | Constructed return/actor-gradient checks pass with 41 bins; no evidence in those checks requires 81 or 161. All scored returns use the revised bounded dense-reward definition. |
| Multi-episode real-time pipeline | Each loss completed three episodes, 44 updates (28 online, 16 stopped drain/sleep), 36 retirements, and 0/44 non-startup fallbacks. Peak label backlog was 17; it drained to zero. These paired untrained pilots are timing acceptance, not evidence of task competence or uninterrupted one-update-per-observation service. |
| Optional retired-data sleep | Implemented and CPU-tested, including exact crash recovery, immutable archive identity and explicit supplemental-rehearsal accounting. Default off; not used by either timing pilot or admitted as a primary treatment. |
| Maintained Pi and speech interfaces | Actual calibrated actuator/camera adapter, causal instruction ledger and bounded speech bridge delivered and tested through injected hardware/recorded inputs. Real servo calibration, speech-model inference, microphone use and physical motion remain unvalidated. |

Fresh 20-layout teacher validation was **not admitted** because the
development full-task readiness guide failed. All failed/partial results
remain in the notebook, including preselected ego/top-down videos. The
longer prompt performed worse than the ported interface; this repair does
not establish that the frozen Qwen model cannot navigate under other
interfaces. Numeric rewards now provide a coarse approach signal, but do
not yet reproduce the desired smoothly graded reward density.

**Measured resource check-in:** version 2 charged 17.9 active minutes; both
Fix 2 attempts charged 21.3 minutes. Including Fix 1, cumulative repairs
charged **31.1 minutes**, within the existing three-hour repair envelope.
The paired pipeline wall times were 84.5 s (control) and 84.2 s (FD), with
median fit wall times near 0.60 s. These include cooperative scheduling
waits and are not a controlled claim that FD is cheaper than the control.
The timing recipe uses context 4, capacity 8, new/replay batches 1/1 and a
one-second physical control period. A durable checkpoint transaction took
0.057 s median; its 4.76 MB initial learner state does not include populated
optimizer moments.

Conditional whole-study forecasts, including earlier acceptance/profiling
and all repair charges, are **21.4--30.8 h** for six pairs at 2,400 new
transitions/arm, or **11.6--17.2 h** for four pairs at 1,200/arm. The sealed
forecast originally charged repairs but omitted 10.6 minutes of earlier
acceptance/profiling; reporting adds that time explicitly without changing
the sealed artifact. These serialized operation estimates include measured
cooperative waits, so they are conservative, but do not establish readiness
or select a production replay/service recipe. Bootstrap/teacher rescue may
change costs. **Execution stops here for the task/resource check-in; no
Phase 2--5, physical actuation or STM efficacy experiment was started.**

**Closeout integrity:** verified all 60 completed Fix 2 units and all three
checksummed source archives without changing completion markers. Completed
campaign resume verified/skipped in under one second without importing Torch.
The final notebook's eight code cells execute artifact-only; 24 focused
reporting tests pass after the reporting-only readability changes. Four
preselected videos decode with the recorded frame counts and visible motion.
Fix 1's section and `mathematical_overview.ipynb` retain their prior hashes.

**Follow-up recognition diagnostic (2026-11-10, user requested):** frozen
Qwen answered six independent questions on four saved initial frames (seeds
601/611, with/without the obstacle) and one ball-absent control. All 30 raw
answers and input/model provenance are sealed separately under
`cache/numerical_experiment/picar/diagnostics/visual-questions-v1`; active
compute was 89.3 seconds. No control prompt or prior answer primed a query.
The open-ended question spontaneously identified the red sphere in all four
positive images. Direct ball/sphere questions correctly rejected the absent
control, but the two distance questions invented a sphere position there.
Obstructed views were often described as a ball "on top of" the block, and
distance replies used unsupported units/coordinates. This argues against
basic recognition failing solely because of the synthetic appearance on
these images; it does not establish reliable depth, navigation or use of
vision under the control prompt. No existing run, task or readiness guide
was changed, and no later learning phase was started.

#### Phase 1 Fix 3: Centering Rewards And A Gradual Navigation Curriculum

**Motivation and status (2026-11-11):** agreed design, drafted for review;
implementation and execution are pending. Fix 2 produced useful straight
approaches (3/6 unobstructed successes with the ported teacher), but no
obstacle-task successes. Its follow-up recognition diagnostic identified the
red sphere in every positive image while exposing unreliable spatial and
distance descriptions. These findings motivate giving the **trainable policy**
an easier starting distribution and a gradually increasing need to navigate.
They do not establish that recognition is universally reliable or that the
frozen teacher can acquire spatial skills from additional rewards.

Qwen's demonstrator and evaluator remain frozen. Qwen supplies initial
experience, followed by one-time student distillation; the student then learns
from its own experience through the existing joint-likelihood update. The
purpose of this amendment is to make that progression viable, not require
the demonstrator to solve the eventual difficult task before learning starts.
Preserve Fixes 1 and 2, their prompts, execution records and artifacts.

**Scope and precedence:** version the common visual reward, navigation prompt
and reset distribution. Keep `ball-contact-v1`, the eight-action weighted
control mapping, top-$L$ architecture, STM/LTM parameterization and eviction-only
diagonal EWC. For this revision, the curriculum and easier-task bootstrap
below supersede Fix 2's requirement that frozen Qwen first pass full-obstacle
readiness, and the original requirement to keep obstacle geometry fixed.
The later Phase 2 skill/readiness checks apply to the admitted **initial easy
distribution**, not the final obstacle distribution. Difficult-task competence
is something the student is intended to learn. No claim of STM efficacy or
full Phase 2--5 execution follows from this repair alone.

**1. Add a small, auditable centering reward.**

Keep proximity and terminal success dominant. Use one frozen-Qwen evaluation
of each post-action image to obtain visible-ball status $V\in\{0,1\}$,
proximity $p\in[0,10]$, and horizontal centering $c\in[0,1]$. The centering
rubric should award 1 for roughly central placement and decrease smoothly
toward 0 near the image edges; it must not penalize ordinary small offsets.
Keep the semantic rubric short, with the strict output schema specified
separately. Preserve raw responses and validate every component. These are
visual judgments, not calibrated probabilities or geometric measurements.

Use the recorded camera pan $\phi$, available through the shared robot
interface, to discount centering obtained by looking sideways:

$$
w(\phi)=\max\!\left(0,1-\frac{|\phi|}{10^\circ}\right),
\qquad
v=V\min\{10,\;p+\eta w(\phi)c\},\qquad \eta=0.5.
$$

Use consistent angle units in code. This is at most half a point of bonus on
the existing 0--10 visual scale; camera pan alone must not earn the full
bonus. It is a practical mitigation, not a proof against every reward exploit.
Qwen receives RGB for its scores, not simulator target position, distance,
pose or the intended panel label. Pan affects only the transparent arithmetic
above, and must be measured/recorded rather than inferred from a requested
command when the backend can lag. Missing pan is a diagnostic failure, not
an assumed forward view. Store proximity and awarded centering separately.

Feed $v$ into Fix 2's elapsed-time-weighted `dense-visual-v1` formula under a
new reward identity. Keep terminal bonuses, time cost and visual budget
unchanged. The total centering addition is at most 0.1 undiscounted return
units over a 20-second episode with those defaults; the enclosing return
range remains [-6,12]. Near-contact crops retain their proximity reward even
when centering is uncertain. A detour may temporarily lose the small bonus;
there is no rule requiring the ball to remain centered during avoidance.

Construct matched development images at comparable apparent ball size with
central, moderately offset and edge positions. Include left/right mirrors,
absent balls, close crops, obstacle detours and camera-pan-only contrasts.
Show raw component scores and final rewards. Test that proximity ordering
survives, absence gives zero, and central placement usually improves the score
when proximity is matched. Validate the selected rubric on disjoint images;
report ties and failures. Allow one targeted rubric repair, not a broad
prompt search. Check that reward differences survive return binning and
produce different likelihood weights, using Fix 2's bounded resolution audit
if necessary. A requested bonus that disappears in scoring or discretization
is not an implemented learning signal.

**2. Give the teacher concise, consistent guidance.**

Add the following semantic instruction to the versioned navigation prompt:

> When the route is clear, steer toward the red ball and straighten as it
> becomes centered. Prefer keeping it roughly centered while approaching.
> When an obstacle blocks the route, go around it; temporarily moving the
> ball away from the image center is acceptable.

Retain the necessary action/output contract and causal instruction ledger.
Supply the task guidance at every decision, as the existing interface does.
Do not add verbose reasoning requests or require a reasoning transcript.
Prompt revisions may change frozen-Qwen behavior, but reward feedback does
not train its weights. After initial collection and one-time distillation,
ordinary control belongs to the trainable student; neither teacher actions
nor repeated distillation silently re-enter online learning.

**3. Narrow barrier placement gradually while preserving feasible routes.**

Keep the barrier's size and longitudinal coordinate fixed. At each reset,
let $y_{\mathrm{path}}$ be the lateral coordinate of the start-to-target line
at the barrier's longitudinal coordinate. Sample lateral placement from

$$
Y\mid n,\text{layout}\sim
\operatorname{TruncNormal}\!\left(
 y_{\mathrm{path}},\sigma(n)^2;\;y_{\min},y_{\max}\right),
$$

where the physical bounds are fixed across curriculum progress. Use a smooth
scale schedule, for example

$$
\log\sigma(n)=(1-q(r))\log\sigma_{\mathrm{easy}}
                 +q(r)\log\sigma_{\mathrm{hard}},
\qquad q(r)=10r^3-15r^4+6r^5,
$$

with clipped progress $r$ between a declared ramp start and end, and strictly
positive endpoint scales. This implements the agreed broad-to-concentrated
placement idea with fixed support, avoiding moving uniform endpoints. It
does not itself prove LAN or establish that the chosen rate is sufficiently
slow. Start development with an easy hold for the first 10% of online
exposure, a ramp over 10--80%, and a final hard hold. Use at least the existing
200-transition ramp minimum and several episodes; lengthen within the budget
if the observed change concentrates into only a few resets.

Calibrate scales and bounds through cheap geometry draws before VLM rollouts.
Aim provisionally for 40--60% of early layouts to have a clear direct approach
and at least 90% of late layouts to obstruct it. These are distribution-design
guides, not promised success rates. Determine blockage using the vehicle's
finite footprint and a declared clearance margin, not a point-like ray.
Inspect the resulting density and blockage curves; report achievable rates
instead of silently changing the arena, target or barrier dimensions.

Replace the old screen requiring wide passages on **both** sides. Admit
side-offset barriers when at least one route remains feasible, accounting
for turning radius, start/target access, walls and both endpoint steering
gains. Keep this screen independent of policy and curriculum progress.
Screening changes the realized placement distribution: log rejections and
measure blockage on accepted layouts. Check representative paths with the
scripted controller as a geometry diagnostic; do not use its controls as
primary demonstrations. Preserve the old screen for historical run versions.

Placement stays fixed within an episode. Curriculum progress uses cumulative
captured training control intervals, including executed fallbacks, and is
sampled at reset; pauses, sleep and evaluation do not advance it. Do not
advance difficulty on success or slow it for one method after confirmation.
Pair the schedule and named reset RNG streams across methods. Policies can
finish episodes at different exposures, so log actual reset exposures and
realized layouts rather than claiming identical state trajectories.

During initial teacher collection and distillation, hold difficulty at the
easy endpoint and texture/gain at A. For later online trajectories, retain
the existing smooth A-to-B-to-A texture/gain path and add this gradual barrier
curriculum. Label these axes separately: returning texture/gain to A does
**not** restore the initial placement distribution while barriers remain hard.
Report initial-task retention on the fixed easy-A panel, never on that altered
end-of-training distribution. No full factorial drift study is required.

**4. Verify a learning opportunity, then stop for the check-in.**

After unit/CPU acceptance, execute a small ordered diagnostic rather than
requiring zero-shot obstacle mastery:

1. Compare the historical and revised teacher prompts on the same six easy
   development layouts. Preserve successes and failures, action diversity,
   steering corrections and representative clips. Treat the old prompt as a
   bounded diagnostic comparator, not another full learning arm.
2. Collect at most 300 easy-distribution teacher transitions for a diagnostic
   bootstrap and perform one-time student distillation plus the initial
   likelihood fit. Record all experience and costs; do not select only
   successful episodes. Keep failed demonstrations distinguishable for review.
3. Run a bounded Replay/EWC student trainability pilot from that shared
   initialization, with two development seeds and at most 320 new transitions
   per seed. Use the gradual curriculum and declared exploration. Compare
   pre/post checkpoints on the same held-out diagnostic layouts against the
   frozen initialized student. This probes the common learner and does not
   test an FD effect. Shorten optional evaluation before shortening the ramp
   below its minimum; report an incomplete pilot if the remaining budget cannot
   support it.

The earlier 70% teacher/60% student easy-task guides remain provisional
readiness information, not a demand that the teacher already solve the hard
task. Examine task success, collisions, heading/approach progress and action
coverage before interpreting a small binary panel. At most one common
development repair of curriculum rate or the existing exploration/optimizer
settings is allowed, with a new version and recorded rationale. There is no
requirement for statistically significant pilot learning, nor an automatic
rejection of the idea after a noisy short run. A stationary policy or a lost
reward signal should receive an explicit diagnosis before full-budget runs.

Preselect fixed evaluation panels for clear approaches, partial obstruction
and reliably blocked approaches, including mirrored layouts and small target
offsets. Freeze their layouts and reward rubric across checkpoints. Use
disjoint development and later confirmation seeds; evaluated transitions
never enter replay, distillation, EWC or the exposure clock. Keep the initial
easy-A retention panel and hard-B acquisition panel explicit in later phase
calibration. Training success alone mixes learning with changing difficulty.
After any development repair, both later methods receive the same sealed
curriculum, reward, bootstrap, evaluation and service schedule.

**5. Reporting, recovery and resource acceptance.**

Add a concise, chronological **Phase 1 Fix 3** notebook section with the
motivation above, one settings table and the following reader-facing results:

- Placement density at early/middle/late exposure, with direct-path blockage
  probability and the texture/gain schedule clearly distinguished.
- A small reward contact sheet showing proximity, centering and total score,
  including a camera-pan contrast.
- Fixed-panel success/collision rates before and after the student pilot,
  with counts and separate difficulty levels; one representative synchronized
  ego/top-down clip before and after learning on a preselected common layout.
- Progress near the notebook top: completed/pending units, active compute,
  remaining estimate, and a short readiness conclusion with unresolved issues.

Use a new configuration-driven Fix 3 entry point and immutable campaign root
under `cache/numerical_experiment/picar/phase1/`. Preserve old loaders and
completed artifacts. Implement `--resume` before GPU execution, including
placement RNG, exposure/reset state, evaluator cache identity, replay and
retirement ledgers, learner/optimizer state and interrupted-episode handling.
Never turn an interrupted episode into a failure reward. Test reward bounds,
camera-pan discount, collision precedence, distribution/screen invariants,
evaluation isolation and stop/resume equivalence. Record new prompt, reward,
curriculum and source identities. Rescored historical RGB belongs in derived
artifacts with parent hashes; historical returns or Fisher assets cannot be
silently reused under changed labels. Default-off retired-data rehearsal
and the physical robot remain outside this repair.

| Work | Provisional active wall-clock compute |
|---|---:|
| Geometry, focused acceptance and reporting | 0.10--0.20 h |
| Frozen reward/prompt panels and teacher rollouts | 0.15--0.40 h |
| Small bootstrap, student pilot and fixed evaluations | 0.55--1.40 h |
| **Fix 3 total** | **0.80--2.00 h** |

These are estimates, excluding implementation labor, not a fresh allowance.
Charge all attempts to the existing three-hour cumulative repair envelope
and the overall 20-hour study ledger, including Fixes 1/2 and the 89.3-second
recognition diagnostic. Reforecast after the first measured units; preserve
partial results and report any needed extension before exceeding the cap.
At closeout, state what an easy bootstrap can supply, what the student has
actually learned, the admitted difficulty/reward settings and updated costs
for Phases 2--5. Execution of this amendment ends at that check-in; subsequent
full learning and physical operation require their own authorization.

### Phase 2: Sol Bootstrap And Student Readiness

**Purpose:** give the continual learner a real initial capability to retain.

Collect Qwen-controlled A data, perform one-time distillation, fit the initial
joint model, and construct the nonoverlapping initial replay/archive split.
Audit held-out A performance and language responsiveness. Provisional readiness
guides are at least 70% teacher success and 60% student success on a 20-layout
development panel, with the student within 15 percentage points of the teacher.
These are engineering guides, not hypotheses to test for significance.

Allow two bounded rescue attempts using clearer observation history,
demonstration coverage, or a common optimizer/model configuration. Do not
inject privileged geometric actions into the primary Qwen dataset as an
unreported shortcut. If Qwen cannot demonstrate a useful capability, report
that blocker instead of measuring "retention" of an unlearned task.

Confirm the initial controller is meaningfully worse at B and that an
unregularized or lightly regularized development learner can improve there.
Differences of roughly 10 percentage points provide a useful first guide,
not a mandatory post hoc condition for selecting a favorable STM environment.
Use continuous terminal errors to distinguish a binary-metric ceiling from
no adaptation requirement. Freeze common initialization and data provenance.

**Exit:** useful A skill, plausible B learning opportunity, valid initial
likelihood/Fisher assets, and a revised fresh-run forecast.

### Phase 3: Sol Bounded Development And Retention Matching

**Purpose:** calibrate both methods fairly, not assume that MNIST strengths,
ridge scales, or solve budgets transfer to a VLM policy.

Use two development replica seeds disjoint from confirmation. Start with
$\pi_0=.025$, a common short optimizer budget, no implicit weight decay,
and the profiled replay configuration. Tune a common learning-rate/solve
setting before treatment-specific regularization. If the common adaptation
weight is plainly unsuitable, a bounded shared check at .01 and .05 is
allowed and recorded. Do not tune it independently to favor an arm.

Use the task, reward formula and bin resolution admitted by the latest
versioned Phase 1 repair. Recheck return bounds and weight concentration on
the actual Phase 2 bootstrap; old smoke scales are not selected hyperparameters.
Calibrate the common RL scales before comparing FD strengths. Set the initial
temperature from a robust spread of bootstrap returns/advantages, then check
half/base/double temperature only if weight concentration or absent learning
warrants it. Inspect normalized weight concentration and saturation; a
weight-based effective count is a weighting diagnostic, not an iid sample
count. Initialize $\Sigma_A$ from stabilized bootstrap action residual scales
or a documented common isotropic scale, and freeze it after development.
Select exploration $\Gamma$ by observed control variation and safe task
coverage, not logit variance alone: a saturated softmax can make sizeable
logit noise physically ineffective. These are bounded, sequential common
calibrations, not additional treatment-specific factorials. Record all values
and preserve their normalizing terms in likelihood checks.

First obtain a strong Replay/EWC recipe. Explore a small ridge bracket,
initially $c_r\in\{.03,.1,.3\}$; extend to 0 or 1 only if a boundary is clearly
limiting. Then shortlist positive FD strengths using initial penalty-to-data
gradient ratios around .01, .1, and 1, followed by actual trajectories. These
ratios initialize a search, not evidence that the resulting penalties have
equal effects. Validate $\epsilon$ before changing strength.

Use held-out A and B development panels to plot acquisition versus retention
and cost. First require the control to remain within a provisional 10-point
A failure increase from initialization. Match FD's average and endpoint A
failure to the selected control within a provisional five-percentage-point
tolerance, while selecting for B performance. These tolerances must be checked
against panel resolution and uncertainty before being sealed. Repeated panels
improve measurement, not the number of independent learning replicas.

If no matched pair exists, report the frontier and keep the best informative
comparison with its retention qualification; do not erase it or silently
relax tolerances after fresh evaluation. Allow at most two documented
development repair rounds inside the allotted budget. Sol may correct
numerical defects and poor common settings without waiting for the user.

Before confirmation, seal model/role revisions, both recipes, drift duration,
return design, temperature/action scales, context/replay rules, sleep,
evaluation panels, target capability, inference convention, seeds, and analysis.
Decide trajectory length and replica count from measured costs before looking
at fresh outcomes. The starting target is 2,400 unique transitions per arm;
1,200--3,600 is the permitted pre-confirmation sizing bracket, subject to the
ramp-duration and readiness requirements.

**Exit:** a sealed comparison, retention status, capability target, and
resource forecast. Publish the selected and rejected development candidates
in compact artifact-backed form.

### Phase 4: Sol Fresh Paired Learning

**Purpose:** evaluate new-task sample efficiency and retained capability under
the frozen design, not continue hyperparameter tuning on fresh results.

Target six paired replicas. If the profiled 20-hour allocation cannot fit six,
choose four complete pairs before confirmation or request a measured extension.
Do not start an infeasible allocation and later stop when significance looks
convenient. Alternate method order within pairs and prioritize finishing pairs
when the remaining resource allowance becomes small.

Evaluate initialization and post-update checkpoints near exposure fractions
.15, .30, .45, .55, .80, and 1.0. Start the cost forecast with four A and four B
held-out layouts at intermediate checkpoints and twelve of each at the end;
Phase 3 must increase these or weaken the sample-crossing claim if resolution
is inadequate. The exact counts are sealed before fresh outcomes. Evaluation
has its own RNG and does not advance training, mutate recurrent training
state, add replay observations, or update EWC.

Run at realistic control cadence, including deadline behavior for evaluation.
Measure both learning and full operational costs. Save checkpoint videos on
preselected common layout seeds and an explicitly separate failure gallery.
Keep raw outcomes and learning trajectories even if a run is disappointing.

Refresh the notebook after completed pairs or every 15 minutes when meaningful
new artifacts exist. Show counts for completed, incomplete, and remaining
units, elapsed charged hours, current stage, and a range for remaining time.
Intermediate views are for progress and health, not significance-driven stopping.

### Phase 5: Sol Efficiency Analysis And Closeout

**Purpose:** answer the applied question succinctly and leave reusable robotics
infrastructure, regardless of which method wins.

Verify final manifests, pairing, retirement accounting, seeds, and resource
charges. Produce the primary paired effect table, A/B capability curves,
sample-equivalence graphic with censoring, cost table, and representative
synchronized videos. Separate retention-matched conclusions from trade-offs.

The conclusion should state what to do with the method, how many new
transitions it saved where supported, the compute multiplier and $S$, whether
old capability was preserved, and which uncertainty prevents stronger claims.
Discuss frozen-controller and Replay/EWC results without treating a negative
STM outcome as an implementation failure by default.

Deliver a simulation-to-physical calibration checklist and a working fake-driver
Pi deployment test, not an assertion that the real robot was validated. Do
not automatically run a physical demonstration. Preserve code, configs,
initial assets, and run manifests needed to revisit a result from the archive.

## Wall-Clock Budget And Autonomy

These are planning estimates, not measured Qwen timings. Engineering effort
is distinct from experiment runtime; expected Phase 0 coding/debugging is
roughly 6--14 hours of elapsed work and can exceed that if dependency or
hardware issues arise. Do not charge idle user pauses as active compute.
The Phase 0 model split changes who performs that work, not this unmeasured
estimate or the experimental compute allowance. Reforecast engineering time
and model usage from actual progress; do not promise a token-saving percentage
before collecting usage evidence.

| Phase | Active compute forecast | Initial share of 20-hour target |
|---|---:|---:|
| 0: Acceptance and bounded real-model smoke | 0.5--1.5 h | 1.0 h |
| 1: Simulator, roles, and resource calibration | 1--3 h | 1.5 h |
| 2: Teacher collection and student initialization | 2--5 h | 3.0 h |
| 3: Development and retention matching | 3--6 h | 4.0 h |
| 4: Fresh paired trajectories and held-out panels | 8--16 h | 9.5 h |
| 5: Final audit, figures, and closeout | 0.5--1.5 h | 1.0 h |
| **Total** | **15--33 h** | **20 h** |

The target shares are not additional allowances. All active experiment work,
failed attempts, reporting, I/O-bound work, teacher/evaluator inference,
sleep, and acceptance runs count once in a cumulative wall-clock ledger.
Concurrent activities do not each create another full allocation. Track GPU
busy time, operational simulated time, and total elapsed time separately.

After Phase 1 and again after Phase 2, use measured throughput to forecast the
remaining work. Sol may reduce the predeclared sizing choices within the
allowed bracket, narrow optional diagnostics, or redistribute time between
phases. Do not remove the control, language path, retention matching, or
resume requirements to fit a misleading budget. If a sound experiment needs
more than 20 hours, report why, the revised estimate, and what fits without
the extension; wait for approval before exceeding the cap.

Sol may fix implementation/numerical defects, improve scheduling, and choose
development settings without requiring Astra or an available user. Material
changes get a dated notebook note, code/config version, affected artifact IDs,
and rerun decision. Changing the likelihood, control estimand, reward target,
physical safety policy, or confirmed recipe requires a new experimental
version; affected results cannot be pooled as if unchanged. The original
artifacts remain immutable. Budget extensions and physical motion remain
explicit user decisions.

## Human-Friendly Notebook And Video

The notebook should be short enough to read as a research narrative, not an
execution log. Aim for the following visible structure:

1. **Question and progress:** motivation, live stage/status table, elapsed/remaining resources.
2. **Task and setup:** one scene view, drift plot, action mapping, tiny settings table.
3. **Readiness:** initial capability, language checks, and measured latency/VRAM.
4. **Development:** acquisition/retention frontier and selected recipes.
5. **Fresh learning:** B success and A retention versus unique new transitions.
6. **Efficiency:** sample-saving arrow plot, attainment/censoring, compact $f,c,S$ table.
7. **Behavior and mechanisms:** synchronized clips and a few diagnostic summaries.
8. **Conclusion:** applied recommendation, limitations, and transfer readiness.

Keep phases chronological. Put a concise definition beside each unfamiliar
metric; distinguish training returns from held-out performance, online samples
from replay, and observed versus fitted crossings. Use consistent labels and
units, legends for references, uncertainty bands, and visible evaluation
checkpoints. Tables should summarize effects, not contain every logged field.

Detailed per-step logs, model manifests, all trial rows, covariance diagnostics,
and failed-case galleries live in linked artifacts or a secondary diagnostic
report. Do not auto-insert hundreds of configuration or status paragraphs.
The theoretical derivation lives in this plan; the notebook needs a short
explanation and a link, not a duplicate textbook chapter.

Render egocentric camera and top-down vehicle trajectory together, with
timestamps, drift, intended/executed controls, relevant language, and outcome.
Top-down privileged information is for the reader only. Use common preselected
seeds across checkpoints for a fair visual comparison. Clearly label a
separately selected failure gallery so it is not mistaken for a representative
random sample. Small inline previews may link to external full-resolution
clips; missing external media should produce a clear message, not block all
scalar analysis.

Progress reporting may show incomplete studies, but must never treat a
partially written scientific unit as complete or silently repair its data.
Display missing results as pending/unavailable with a reason. Reporting must
not load the VLM, train, sample scores, download data, or rebuild large matrices.

## Organization And Restart Contract

### Proposed Layout

```text
numerical_experiment/picar/
    plan19.md
    plan19_results.ipynb           # created by Phase 0
    PLAN19_HANDOFF.md              # updated at milestones and pauses
    configs/
    run.py                        # stage runner and resume entry point
    report.py                     # artifact-only report generation
    ...                           # task-specific implementation
src/robotics/                     # shared protocol, clients, calibration
src/continual/                    # only genuinely shared learning components
demo/picar/                       # lightweight Pi service and applied entry points
cache/numerical_experiment/picar/plan19/<version>/
```

These are proposed modules, not a requirement to create empty abstractions.
Reuse existing utilities with honest dependencies; extract generic code only
where it actually serves multiple callers. Avoid a robotics runtime dependency
on an entire MNIST experiment package. Do not copy code solely to avoid a
small, legitimate shared import. Document the new experiment namespace in
the repository map.

Provide a standalone command of the following form after implementation:

```bash
~/.venv/bin/python -u -m numerical_experiment.picar.run \
  --config numerical_experiment/picar/configs/plan19.json \
  --root cache/numerical_experiment/picar/plan19 \
  --archive-root "<external-directory>/plan19" \
  --resume
```

This is an interface requirement, not a currently executable command. Support
phase selection, status, report refresh, and graceful stop. The runner must
work without an active assistant or IDE session, including inside `tmux`.
Log readable progress and the next recovery command.

### Immutable Artifacts And Atomic Progress

Each bootstrap, development trajectory, fresh trajectory, evaluation panel,
and final analysis has a versioned manifest with resolved config/hash, role
and model revisions, seeds, code commit/dirty state, software/CUDA/device/dtype,
schema versions, timestamps, completion status, and artifact checksums.
Completed scientific units are append-only. Write temporary payloads, flush,
then atomically commit their manifest and completion marker.

Checkpoint at bounded action/learning transactions, not only at the end of a
multi-hour replica. Include model/adapters, optimizer, EWC diagonal and anchor,
retirement ledger, replay eligibility and context refs, pending returns,
language/recurrent state, all named RNGs, action sequence IDs, training clock,
simulator state, update/label queues, and cumulative resource charges.
Do not acknowledge scientific completion before external payloads are durable.

A process-local PyBullet `saveState` identifier is not a reboot checkpoint.
Implement and test durable restoration or deterministic reconstruction from
an episode boundary and recorded actions, including velocities, controller
state, collision behavior, and randomization. Compare resumed trajectories
within declared numerical tolerances. If exact mid-episode restoration cannot
be supported, preserve the interrupted prefix and restart an explicitly new
episode; do not invent returns or claim bitwise uninterrupted equivalence.
Physical world state cannot be restored by loading a model: physical resumption
requires a fresh operator-approved safe reset.

The eviction transaction must atomically record archival admission, updated
Fisher/anchor, and eligibility removal. A crash may retry an uncommitted
transaction, never admit the same transition twice. Similar ledgers protect
new-experience counts, return labels, and action acknowledgements.

On SIGTERM or a stop request, finish or roll back the current bounded write,
stop issuing motion, checkpoint, refresh status, and exit. Stop checks belong
inside long validation/archive loops too; do not make a user wait an hour for
a read-only report scan. Forced restart loses at most the declared unfinished
transaction, whose status remains visible.

Resume validates config/recipe identity, required schemas, and incremental
artifact manifests. Do full integrity audits at acceptance and closeout,
with interruptible streaming checks; do not rebuild every historical notebook
or reread an entire video archive at every restart. Cache verified immutable
summaries by manifest hash and invalidate them if inputs change.

## Interpretation And Deferred Work

A positive result would establish a useful retention-qualified experience/cost
trade-off for this simulator and this shared initialization, not automatically
for arbitrary real robots or all RL objectives. A negative result can localize
the difficulty to initial competence, action ambiguity, reward support,
retention trade-offs, dependence, or online resource limits. Preserve those
distinctions rather than folding every failure into one success-rate number.

Keep these outside the initial comparison: real-world learning runs, retired
archive replay during sleep, online teacher distillation, adaptive $\pi$, LFUs,
actor mixture models, learned reward-model updates, changing return targets
mid-run, and broad replay/architecture factorials. Each is a plausible later
question, not a hidden fallback that changes what this experiment estimates.

The immediate deliverable is a credible bridge from the small controlled
studies to a maintained robot learning stack: clear likelihood semantics,
working language-to-action conditioning, bounded online computation, realistic
timing, inspectable behavior, and a fair test of STM's practical sample value.

## References

The algebra above is the proposed Plan 19 construction. These sources supply
precedents, assumptions, and implementation guidance, not prior experimental
validation of this exact combination.

1. Peng, Kumar, Zhang, and Levine (2019). [Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning][awr]. Weighted regression as an off-policy policy-learning mechanism.
2. Levine (2018). [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review][inference]. Qualified links between control and probabilistic inference.
3. Abdolmaleki et al. (2018). [Maximum a Posteriori Policy Optimisation][mpo]. Related probabilistic policy-improvement construction.
4. Bellemare, Dabney, and Munos (2017). [A Distributional Perspective on Reinforcement Learning][distributional]. Return-distribution modeling precedent.
5. Kirkpatrick et al. (2017). [Overcoming Catastrophic Forgetting in Neural Networks][ewc]. Fisher-based retention regularization.
6. Schwarz et al. (2018). [Progress & Compress: A Scalable Framework for Continual Learning][progress]. Consolidation precedent, distinct from this plan's one-time distillation.
7. Hinton, Vinyals, and Dean (2015). [Distilling the Knowledge in a Neural Network][distillation]. Predictive-distribution distillation.
8. Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models][lora]. Additive low-rank adaptation.
9. Dettmers et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs][qlora]. Quantized frozen-backbone adaptation.
10. Kapturowski et al. (2019). [Recurrent Experience Replay in Distributed Reinforcement Learning][r2d2]. Recurrent-state staleness and replay context.
11. Ng, Harada, and Russell (1999). [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping][shaping]. Potential-based shaping conditions.
12. Farama Foundation. [Gymnasium: Handling Time Limits][time]. Task termination versus external truncation.
13. Qwen Team. [Qwen2.5-VL-3B-Instruct model card][qwen]. Pin the actual model revision at implementation.
14. Hugging Face. [PEFT LoRA reference][peft]. Layer-selection configuration; verify behavior in the installed version.
15. PyTorch. [Autograd mechanics][autograd]. Gradient recording and frozen-module boundaries.
16. Bullet Physics. [Bullet/PyBullet repository][bullet]. Dynamics and simulator implementation.

[awr]: https://arxiv.org/abs/1910.00177
[inference]: https://arxiv.org/abs/1805.00909
[mpo]: https://arxiv.org/abs/1806.06920
[distributional]: https://proceedings.mlr.press/v70/bellemare17a.html
[ewc]: https://arxiv.org/abs/1612.00796
[progress]: https://proceedings.mlr.press/v80/schwarz18a.html
[distillation]: https://arxiv.org/abs/1503.02531
[lora]: https://arxiv.org/abs/2106.09685
[qlora]: https://arxiv.org/abs/2305.14314
[r2d2]: https://willdabney.com/publication/r2d2/
[shaping]: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
[time]: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
[qwen]: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
[peft]: https://huggingface.co/docs/peft/en/package_reference/lora
[autograd]: https://docs.pytorch.org/docs/stable/notes/autograd.html
[bullet]: https://github.com/bulletphysics/bullet3
[openai-models]: https://developers.openai.com/api/docs/models
[openai-selection]: https://learn.chatgpt.com/docs/model-selection
