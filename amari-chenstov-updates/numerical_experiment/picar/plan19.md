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
Phases 1--5 and physical robot motion have not been started. Full-workload
resource admission, task readiness and scientific stage orchestration remain
later-phase work. See `PLAN19_HANDOFF.md` for results and recovery notes. The theory below is
provisional and belongs here until
experiments establish its value. Do not modify
[`mathematical_overview.ipynb`](../../mathematical_overview.ipynb).

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

Reusing already retired observations during sleep is an implemented,
default-off option. Enabling it is a separately versioned experimental
condition, requiring an explicit accounting rule for repeated evidence and
archived EWC; it is not an autonomous performance rescue in the primary pair.

## Simulator And Continuous Drift

### Task And Geometry

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
