# Implementation Plan 1: MNIST LFU Experiment

This plan turns [AGENTS.md](AGENTS.md) into an implementation sequence. It is deliberately phase-gated: each phase produces working software and scientific evidence, then pauses for a user check-in before the next phase begins.

The approach is Waterfall-inspired in dependency order and Agile in revision. Earlier phases establish contracts that later work depends on, while every check-in may revise the remaining phases in response to evidence. Do not begin the next phase until the current phase has been reviewed and explicitly accepted.

## Governing objective

Build a reproducible experiment that determines whether the full linearized Fisher update

$$
D\mathcal I_\theta[u]=(C_\theta+R_\theta):u
$$

tracks the unique likelihood Fisher more accurately than EMA or an Amari-Chentsov-only update, and whether any tracking improvement benefits EWC continual learning as

$$
p\longmapsto\theta^\star(p)\longmapsto\mathcal I(\theta^\star(p))
$$

moves from $p=0$ to $p=1$.

The implementation must support immutable paired replicas, dense reference calculations, diagonal and low-rank-plus-diagonal approximations, and a final controller experiment using the mixture-derived EWC objective

$$
\widehat\theta_{t+1}
=
\arg\min_\vartheta
\left\{
\overline L_{\mathrm{new},t}(\vartheta)
+
\frac{1-\pi_t}{2\pi_t}
(\vartheta-\widehat\theta_t)^T
\widehat{\mathcal I}_t
(\vartheta-\widehat\theta_t)
\right\}.
$$

The realized displacement is
$u_t=\widehat\theta_{t+1}-\widehat\theta_t$; it is not multiplied by
$\pi_t$ after optimization.

## Phase-gate protocol

Every phase ends with a check-in containing:

1. a concise implementation summary;
2. the exact tests and commands run;
3. links to representative artifacts or plots;
4. observed numerical behavior and performance;
5. deviations from this plan;
6. unresolved risks and proposed adjustments;
7. an explicit recommendation to continue, revise, repeat, or stop.

Completed phases should not be casually redesigned. If later evidence invalidates an earlier assumption, open a corrective phase, preserve the original artifacts, and record why the contract changed.

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Foundations and run contracts | Complete |
| 1 | Mathematical kernels | Complete |
| 2 | Canonical model, data, and initialization | Complete |
| 3 | Reference Fisher and stencil validation | Complete |
| 4 | Dense fixed-trajectory experiment | Complete |
| 5 | Results notebook and factor pilot | Complete |
| 6 | Dense EWC-coupled experiment | Complete |
| 7 | Diagonal and low-rank-plus-diagonal representations | Complete |
| 8 | Optimal-controller experiment | In progress |
| 9 | Replication, hardening, and handoff | Pending |

## Cross-cutting constraints

- Computationally expensive operations run in Python entry points, never in the results notebook.
- Long GPU experiments require explicit user approval. Development uses unit tests and tiny smoke configurations.
- All formulas in code identify whether they use scores $s$ or loss gradients $g=-s$.
- All Fisher and LFU statistics use per-sample gradients.
- HVPs are matrix-free. Never materialize a per-sample Hessian.
- Dense spectral calculations and accumulated Fisher statistics use `float64` unless a precision experiment says otherwise.
- Every stochastic component has a separately named and recorded seed.
- Completed run directories are immutable.
- The copied `src/lanczos.py` remains behaviorally unchanged; integration occurs through a wrapper.
- The implementation must not silently launch a full Cartesian sweep.

### Environmental mixture and adaptation weight

Keep the environmental path and the algorithmic evidence weight distinct:

- $p_t$ controls the digit mixture that generates the observations and hence
  the moving target $\theta^\star(p_t)$;
- $\pi_t$ is the effective new-data weight used by the EWC optimization.

For a Bernoulli old/new observation label with
$\pi_t\approx n_{\mathrm{new},t}/n_t$, division of the pooled approximate
negative log likelihood by $n_t$ gives

$$
\pi_t\overline L_{\mathrm{new},t}(\vartheta)
+
\frac{1-\pi_t}{2}
(\vartheta-\widehat\theta_t)^T
\widehat{\mathcal I}_t
(\vartheta-\widehat\theta_t).
$$

For $\pi_t>0$, multiplying this objective by $1/\pi_t$ gives the numerically
convenient odds form used in code,

$$
\overline L_{\mathrm{new},t}(\vartheta)
+
\frac{\lambda_t}{2}
(\vartheta-\widehat\theta_t)^T
\widehat{\mathcal I}_t
(\vartheta-\widehat\theta_t),
\qquad
\lambda_t=\frac{1-\pi_t}{\pi_t}.
$$

The two forms have the same exact optimizer. Because every practical estimate
is early stopped, the implementation records the chosen normalization,
learning rate, and inner-step count. It uses the odds form so the configured
learning rate continues to scale the mean new-data loss directly.

Choosing or capping $\pi_t$ independently of literal sample proportions changes
the effective old-to-new evidence ratio. This can bias the estimate relative to
the instantaneous MLE $\theta^\star(p_t)$ and produce lag under a moving
environment. The intended trade is lower variance, retained old-task
performance, and protection against movements that invalidate the compressed
quadratic or first-order LFU. Evaluate this choice through tracking,
retention, adaptation, and paired trajectory outcomes rather than claiming
ordinary stationary-model consistency.

---

## Phase 0: Foundations and run contracts

### Goal

Create the smallest stable software and artifact foundation needed by every later phase.

### Scope

Phase 6 selected EMA as the primary low-compute control, periodic fresh as the
expensive reference-like condition, ridge full LFU as the principal LFU
treatment, and raw full LFU as an instability ablation. Phase 7 evaluates
representation loss first on a fixed ridge-full trajectory. A structured
representation may enter a coupled run only after it improves materially over
the diagonal boundary without unacceptable instability.

1. Audit the local Python, PyTorch, torchvision, CUDA, NumPy, plotting, and notebook environment. Record available versions; do not install or upgrade dependencies without approval.
2. Establish a minimal importable package layout under `src/`. Prefer a small number of cohesive modules:
   - parameter layout and vector operations;
   - derivative and LFU kernels;
   - Fisher representations;
   - MNIST model and stream generation;
   - EWC and controllers;
   - experiment configuration and artifacts;
   - metrics;
   - a wrapper around `src/lanczos.py`.
3. Define a typed, serializable experiment configuration with schema versioning. JSON should be sufficient unless the existing environment strongly favors another format.
4. Define stable component seeds derived from a replica seed:
   - initialization;
   - initialization data;
   - online stream;
   - reference stream;
   - stencil probes;
   - Lanczos/randomized linear algebra.
5. Define the immutable run lifecycle:
   - resolve and validate configuration;
   - calculate a stable configuration hash;
   - create an incomplete temporary run;
   - write manifest and artifacts atomically;
   - create `COMPLETED` only after validation;
   - refuse to overwrite a completed run.
6. Add cache paths and generated artifacts to `.gitignore`.
7. Add one smoke configuration containing tiny values for every required field.

### Planned artifacts

- package initialization under `src/`;
- configuration and seed modules;
- artifact/run-directory module;
- `mnist_experiment/configs/smoke.json`;
- initial unit tests for hashing, seeds, manifests, and immutability.

### Verification gate

- The same semantic configuration produces the same hash regardless of key order.
- Different replica seeds produce distinct, reproducible component seeds.
- An incomplete run can be resumed according to policy.
- A completed run cannot be overwritten.
- Manifests record git and runtime metadata.
- Unit tests run without downloading MNIST or requiring a GPU.

### Check-in decisions

- Confirm the package and artifact layout before mathematical modules depend on it.
- Confirm the configuration format and run-ID convention.
- Confirm which optional dependencies are acceptable.

---

## Phase 1: Mathematical kernels

### Goal

Implement and verify the mathematical primitives independently of MNIST and experimental orchestration.

### Scope

1. Implement a stable `ParameterLayout` that:
   - records parameter names, shapes, offsets, dtype, and total dimension;
   - flattens parameter tuples and restores vectors without changing ordering;
   - validates checkpoints and matrix artifacts against the layout.
2. Implement per-sample negative-log-likelihood gradients.
3. Implement Hessian-vector products by differentiating $g^Tu$ with higher-order autodiff.
4. Implement batch estimates:

   $$
   Z=\frac1m\sum_b g_bg_b^T,
   $$

   $$
   \widehat\Delta_C
   =
   -\frac1m\sum_b(u^Tg_b)g_bg_b^T,
   $$

   $$
   \widehat\Delta_R
   =
   \frac1m\sum_b(H_bg_b^T+g_bH_b^T),
   $$

   $$
   \widehat\Delta_{\mathrm{LFU}}
   =
   \widehat\Delta_C+\widehat\Delta_R.
   $$

5. Implement the one-sample signed factorization

   $$
   U=[g,H_u],
   \qquad
   B=
   \begin{bmatrix}
   -(u^Tg)&1\\
   1&0
   \end{bmatrix},
   \qquad
   UBU^T=\widehat\Delta_{\mathrm{LFU}}.
   $$

6. Implement dense symmetrization and Frobenius PSD projection, returning both the projected matrix and pre-projection diagnostics.
7. Implement lightweight matrix-free applications of $Z$, $\widehat\Delta_C$, and $\widehat\Delta_R$ to probe vectors. Later representations will reuse these operators.

### Tests

- flatten/restore round trips for heterogeneous parameter shapes;
- layout mismatch detection;
- score equals negative loss gradient;
- HVP agrees with a central finite difference of gradients;
- explicit LFU equals $UBU^T$;
- batched estimates equal the average of one-sample estimates;
- AC and residual signs agree with the mathematical overview;
- projected matrices are symmetric PSD;
- Frobenius PSD projection does not increase distance to a known PSD target;
- dense and matrix-free products agree.

Use tiny smooth toy likelihoods in `float64`. Include at least one canonical case where $R$ should vanish in expectation and one noncanonical parameterization where it should not.

### Verification gate

- Exact algebraic tests pass near floating-point precision.
- HVP relative error is stable across a small finite-difference step grid.
- The noncanonical toy problem produces a resolvable residual signal.
- CPU and GPU results agree within documented tolerances when CUDA is available.

### Check-in decisions

- Confirm autodiff time and memory are acceptable.
- Select the per-sample gradient strategy, such as `torch.func.vmap` or a verified loop fallback.
- Confirm tolerances before using these kernels as scientific measurements.

### Decision record

Phase 1 was accepted on 2026-07-28. Use `torch.func.vmap` as the production
per-sample gradient and HVP strategy, while retaining the explicit loop as the
correctness reference and fallback. On the agreed 512-parameter CNN with an
eight-observation `float64` batch, `vmap` took 7.6 ms on the RTX 4070 versus
16.7 ms for the loop. Peak allocated GPU memory increased from 16.7 MiB to
24.1 MiB. CPU timings were 7.5 ms for `vmap` and 8.5 ms for the loop.

---

## Phase 2: Canonical model, data, and initialization

### Goal

Implement the agreed 512-parameter CNN, reproducible MNIST mixture trajectories, and shared $p=0$ initialization artifacts.

### Scope

1. Implement the canonical network:
   - `Conv2d(1, 4, 3, padding=1)`;
   - `SiLU`;
   - `AvgPool2d(2)`;
   - `Conv2d(4, 6, 3, padding=1)`;
   - `SiLU`;
   - `AdaptiveAvgPool2d((2, 2))`;
   - `Linear(24, 10)`.
2. Assert exactly 512 trainable parameters and register the full-network `ParameterLayout`.
3. Partition MNIST reproducibly into disjoint:
   - initialization data;
   - online trajectory data;
   - reference-Fisher holdout;
   - evaluation data.
4. Implement a mixture sampler with

   $$
   p_t=\frac{t}{K-1},
   $$

   where `num_p_steps` and `samples_per_step` are required configuration values.
5. Keep the conditional distribution of digits 0 through 8 fixed. Support empirical-frequency and balanced modes, with empirical frequency as the canonical default.
6. Materialize observation identifiers and class choices for a replica before running treatment conditions. Store this stream plan so paired conditions consume identical observations.
7. Implement high-quality $p=0$ fitting and save:
   - model parameters;
   - parameter layout;
   - optimizer-independent metadata;
   - evaluation metrics;
   - the data and seed identities needed to reproduce initialization.

### Tests

- exact parameter count and output shape;
- no forbidden modules such as dropout or batch normalization;
- deterministic initialization by seed;
- deterministic mixture-stream generation;
- empirical digit-9 frequency converges to configured $p$;
- the non-9 conditional class distribution remains fixed across $p$;
- data partitions are disjoint;
- all paired conditions load identical initial parameters and stream plans.

### Verification gate

- A tiny CPU initialization and trajectory sampler completes.
- A GPU initialization run completes when CUDA is available.
- The canonical model learns a nontrivial MNIST classifier at $p=0$.
- Initialization artifacts reload exactly and preserve parameter ordering.

### Check-in decisions

- Review initial accuracy and class-9 behavior before accepting the architecture.
- Confirm initialization sample size, optimizer, stopping rule, and dtype.
- Confirm the canonical non-9 sampling mode.

### Decision record

Phase 2 was accepted on 2026-07-28. The canonical initialization uses 30,000
$p=0$ observations, Adam with learning rate 0.01, batches of 256, `float32`,
and early stopping at 80% non-9 evaluation accuracy with a 15-epoch maximum.
The canonical non-9 sampler uses empirical conditional MNIST frequencies.

Every replica independently samples its partitions and fits a fresh
early-stopped $p=0$ estimate from its named seeds. MNIST observations may
overlap across replicas. Only paired treatment conditions within the same
replica reuse an exact initialization and stream bundle.

CPU initialization and temporary validation trajectories use deterministic
PyTorch algorithms. When initialization, score, or HVP work is explicitly
configured for CUDA, use deterministic warning mode because PyTorch does not
provide a deterministic CUDA backward implementation for adaptive average
pooling. The exception is emitted visibly and recorded in run metadata. Each
replica's resulting initialization checkpoint is still generated once from its
named seeds and reused exactly across paired conditions; completed matrix and
trajectory artifacts remain content-addressed and immutable.

---

## Phase 3: Reference Fisher and stencil validation

### Goal

Establish a score-only measurement oracle accurate enough to evaluate LFU predictions.

### Scope

1. Implement chunked high-sample reference Fisher estimation:

   $$
   \widehat{\mathcal I}_{\mathrm{ref}}(\theta)
   =
   \frac1{N_{\mathrm{ref}}}
   \sum_{i=1}^{N_{\mathrm{ref}}}
   g_i(\theta)g_i(\theta)^T.
   $$

2. Accumulate in `float64` without retaining all per-sample gradients.
3. Implement content-addressed reference caching keyed by:
   - checkpoint hash;
   - parameter layout;
   - reference stream identity;
   - $p$;
   - sample count;
   - dtype and schema.
4. Implement paired nested-sample convergence diagnostics for increasing `reference_sample_size`.
5. At one or more interior $p_\star$, implement the score-only central stencil

   $$
   \widehat{D\mathcal I}_{\theta_\star}^{\,\mathrm{stencil}}[v]
   =
   \frac{
   \widehat{\mathcal I}_{\mathrm{ref}}(\theta_\star+\epsilon v)
   -
   \widehat{\mathcal I}_{\mathrm{ref}}(\theta_\star-\epsilon v)
   }{2\epsilon}.
   $$

6. Use identical holdout observations at both stencil points.
   To estimate the full Fisher derivative while retaining this pairing, weight
   each perturbed score outer product by the raw likelihood ratio

   $$
   \frac{f_{\theta\pm\epsilon v}(X_i)}{f_\theta(X_i)}.
   $$

   Without this change of measure, the fixed-observation stencil differentiates
   only the score outer product and targets $R:v$, not $(C+R):v$. Retain that
   unweighted stencil as an explicit residual-only diagnostic.
7. Compare the stencil with high-sample AC-only and full LFU estimates across:
   - an epsilon grid;
   - increasing reference sample sizes;
   - several normalized directions;
   - interior $p$ values away from 0 and 1.
8. Benchmark score, HVP, reference-Fisher, and eigendecomposition costs.

### Tests

- chunked and unchunked reference accumulation agree on a tiny problem;
- cache keys change when any scientific input changes;
- cached matrices reject parameter-layout mismatches;
- paired stencils are deterministic under fixed seeds;
- stencil sign and scale agree with an analytically tractable toy likelihood.

### Verification gate

- Reference estimates exhibit a visible convergence regime as sample size increases.
- A nonempty epsilon interval is not dominated by Monte Carlo noise or truncation.
- Full LFU agrees with the stable stencil within uncertainty on the validation cases.
- AC-only and residual contributions are recorded separately, regardless of which is better.
- Measured compute establishes a feasible default reference sample size and checkpoint cadence.

Failure to find a stable stencil is a stop-and-revise event, not permission to proceed with an unverified oracle.

### Check-in decisions

- Select the default and maximum `reference_sample_size`.
- Select the stencil epsilon grid and interior $p$ checkpoints.
- Decide whether reference computation is affordable at every planned checkpoint or must be sparse.
- Revisit model dimension only if the measured costs are prohibitive.

### Decision record

Phase 3 was accepted on 2026-07-28. Use 32,768 observations for routine
reference Fishers and 65,536 observations for maximum-quality checks at
$p\in\{0.25,0.5,0.75\}$. The canonical stencil scale is
$\epsilon=3\times10^{-4}$, with $10^{-4}$ and $10^{-3}$ retained as
sensitivity values.

The paired weighted stencil targets the full derivative $(C+R):v$ and agreed
with the full LFU throughout the stable epsilon range. The unweighted paired
stencil targets $R:v$ and is retained as a residual-only assumption check.
Reference computation is affordable at every planned dense checkpoint.

Phase 4 uses an oracle-assisted, treatment-independent trajectory driver.
`periodic_fresh` uses EMA between fresh reference-Fisher replacements. Dense
updates are projected after every step, with unprojected diagnostics retained.
CUDA score and HVP calculations remain acceptable only if their measured
repeatability error is no more than roughly 10% of reference-Fisher Monte Carlo
disagreement and does not alter treatment rankings.

---

## Phase 4: Dense fixed-trajectory experiment

### Goal

Build the first complete experiment while keeping the parameter trajectory independent of the Fisher estimator being evaluated.

### Scope

1. Implement a treatment-independent trajectory driver. Its configuration, Fisher refresh policy, optimizer, and accepted updates must be recorded.
2. Save an immutable trajectory containing:
   - every $\theta_t$;
   - every realized $u_t$;
   - $p_t$;
   - observation identifiers;
   - optimizer and EWC-driver metadata.
3. Replay the same trajectory through:
   - `ema`;
   - `ac_only`;
   - `full_lfu`;
   - `periodic_fresh`.
4. Enforce the lagged schedule:
   - observations at $\theta_t$ estimate the correction for $u_{t-1}$;
   - the unavailable correction at $t=0$ is zero.
5. Apply the dense candidate update

   $$
   A_t
   =
   (1-\alpha_t)
   \left(
   \widehat{\mathcal I}_{t-1}+\widehat\Delta_t
   \right)
   +\alpha_tZ_t,
   $$

   then symmetrize and project once.
6. Calculate and store all dense tracking and PSD-projection metrics.
7. Integrate the immutable run engine into `mnist_experiment/run_experiment.py`.
8. Add a tiny CPU end-to-end configuration and a small GPU pilot configuration.
9. Audit CUDA repeatability on a short matched trajectory:
   - generate and persist the trajectory once, then replay its exact parameters,
     displacements, and observations for every Fisher condition;
   - repeat the same GPU score, HVP, and LFU calculations from the same inputs;
   - compare a small matched prefix with CPU calculations;
   - report variation in Fisher and trajectory metrics relative to reference-Fisher
     Monte Carlo error and the observed differences between treatments;
   - retain GPU execution when this variation is scientifically negligible, and
     move only the sensitive operation or trajectory generation to CPU if it is
     material.
10. Add amplitude-safe exponentially weighted directional ridge conditions:
    - retain raw `ac_only` and `full_lfu` as controls;
    - add `ridge_ac_only` and `ridge_full_lfu`;
    - estimate the local directional derivatives with separate AC and residual
      numerators and one shared amplitude denominator;
    - cold-start from a zero-derivative prior whose amplitude-scaled
      pseudoinformation decays over the configured half-life;
    - apply zero correction for numerically negligible moves;
    - reset the directional state when signed one-dimensional coherence falls
      below the configured threshold;
    - preserve signed amplitudes so reversal along the same local line does not
      force a reset.

### Tests

- replayed conditions receive byte-identical trajectory and observation identities;
- estimator choice cannot mutate the fixed trajectory;
- `ema` applies no LFU contribution;
- `ac_only` excludes residual terms;
- `full_lfu` includes both terms exactly once;
- periodic refresh occurs at configured indices;
- lagged indexing uses the accepted $u_{t-1}$;
- raw and projected diagnostics are both persisted;
- the GPU repeatability audit uses identical materialized inputs across repeats;
- the CPU/GPU audit reports numerical differences instead of requiring bitwise
  CUDA determinism;
- ridge cold starts attenuate the first correction rather than bias-correcting
  it back to full weight;
- ridge updates remain finite and apply zero correction as $a_t\to0$;
- signed direction reversals are retained while incoherent directions reset;
- AC and residual ridge states remain separately observable;
- the smoke run creates a valid immutable artifact and refuses overwrite.

### Verification gate

- One complete smoke replica runs from $p=0$ to $p=1$.
- All six raw and ridge conditions produce aligned, loadable trajectories.
- Reference comparisons use the same parameter checkpoints.
- No estimator has unexplained NaNs, asymmetry, or unrecorded projection.
- CUDA repeatability error is quantified and judged against explicit scientific
  tolerances before the GPU pilot is accepted.
- Ridge conditions expose warm-up mass, amplitude, coherence, orthogonal
  residual, reset events, denominator, and raw versus smoothed correction
  norms at every step.
- Runtime and storage estimates are available for a realistic pilot.

### Check-in decisions

- Confirm the trajectory driver is scientifically neutral enough.
- Confirm the EMA parameterization and periodic-fresh cadence.
- Decide whether dense PSD projection occurs every step or at a justified configured cadence based on measured cost.
- Approve the pilot grid for `num_p_steps` and `samples_per_step`.

### Corrective ridge extension

Phase 4 was reopened on 2026-07-30 after the first tracking pilot showed that
128-observation AC and residual corrections were too noisy to apply without
temporal pooling. For a locally coherent direction

$$
u_{t-1}=a_tv^{\mathrm{ref}},
$$

the extension maintains

$$
S_{C,t}=\rho S_{C,t-1}+(1-\rho)a_t\widehat\Delta_{C,t},
$$

$$
S_{R,t}=\rho S_{R,t-1}+(1-\rho)a_t\widehat\Delta_{R,t},
$$

$$
q_t=\rho q_{t-1}+(1-\rho)a_t^2,
$$

and applies

$$
\overline\Delta_t
=
a_t\frac{S_{C,t}+S_{R,t}}{q_t+\varepsilon_a^2}.
$$

At the first eligible move in a segment, set $S_C=S_R=0$ and initialize the
pre-update denominator to $q=a_t^2$. The first applied correction is therefore

$$
(1-\rho)
\frac{a_t^2}{a_t^2+\varepsilon_a^2}
\widehat\Delta_t,
$$

not the full single-observation correction. Do not divide by the nominal
warm-up mass $1-\rho^k$: the conservative zero-derivative prior is deliberate.
If $\|u_{t-1}\|\leq\varepsilon_a$, apply zero correction without rotating the
reference direction. On a coherence reset, begin a new cold-start segment.

### Implementation record

Phase 4 reached its check-in on 2026-07-28. The implementation provides:

- an oracle-assisted EWC trajectory driver that materializes every
  $\theta_t$, realized $u_t$, observation identity, reference refresh, and
  proposal metric before estimator replay;
- one shared score/HVP calculation per step, replayed through EMA, AC-only,
  full LFU, and periodic-fresh conditions;
- amplitude-safe directional ridge replay through `ridge_ac_only` and
  `ridge_full_lfu`, with a deliberate zero-derivative cold-start prior,
  separate AC and residual numerators, one shared amplitude denominator,
  signed reversal support, no-motion handling, and coherence resets;
- exact lagged indexing with zero correction at $t=0$;
- dense projection after every complete update, with all raw diagnostics
  retained;
- immutable scalar, trajectory, reference-plan, and selected matrix-checkpoint
  artifacts;
- a full-sequence CUDA repeatability audit and a matched CPU/GPU prefix.

The real-MNIST CPU smoke completed all four aligned conditions and refused a
second invocation after its `COMPLETED` marker was written. The first realistic
GPU pilot exposed an unstable oracle driver: SGD learning rate 0.01 exceeded
the quadratic stability scale near a reference-Fisher leading eigenvalue of
approximately 396. Mean and maximum displacement norms reached 0.285 and
1.186, and AC/full updates sometimes required almost complete PSD projection.
That immutable run is retained as a failed-regime diagnostic.

A paired pilot changing only the driver learning rate to 0.001 removed the
pathology. Mean and maximum displacement norms fell to 0.0264 and 0.0395; the
largest full-LFU relative projection distance was $2.03\times10^{-6}$. Its CUDA
recalculation was bitwise equal across all stored Fisher components, and the
matched CPU/GPU full-LFU relative difference was $4.21\times10^{-15}$. Both
were well below the predeclared $6.04\times10^{-4}$ tolerance derived from a
0.604% nested reference-Fisher disagreement. Treatment ranking was unchanged.

On this single 21-step, 128-observation trajectory, mean relative tracking
errors were 0.167 for periodic fresh, 0.274 for EMA, 1.733 for full LFU, and
1.762 for AC-only. This is evidence that the current online third-moment
correction is too noisy or the path is still too coarse, not a population
comparison: Phase 5 must vary both `num_p_steps` and `samples_per_step` with
paired replicas. Cross-step reference increments also reflect finite tracking
of the implicit $\theta^\star(p)$ path under model misspecification; Phase 3's
fixed-checkpoint stencil remains the direct derivative-identity validation.

The 32,768-score references took about 1.24 seconds each. Trajectory generation
took 7.59 seconds, dense replay took 4.15 seconds, and peak allocated CUDA
memory was about 435 MiB. Selected dense matrix checkpoints make each
realistic run approximately 194 MiB before its content-addressed reference
cache entries.

The corrective ridge CPU smoke completed all six aligned conditions on the
same trajectory hash as the original smoke, persisted the directional state at
selected checkpoints, and refused overwrite after completion. The realistic
ridge pilot likewise reproduced the exact small-step trajectory hash. With an
8-step half-life, amplitude floor $10^{-6}$, and coherence threshold 0.75,
there was one early reset followed by a 19-update segment; median observed
coherence was 0.870 and the final warm-up mass was 0.807.

Directional pooling improved the noisy LFU conditions but did not make them
competitive with EMA on this trajectory. Mean relative tracking error fell
from 1.762 to 1.241 for AC-only and from 1.733 to 1.220 for full LFU. The
periodic-fresh and EMA controls remained at 0.167 and 0.274. Mean applied
full-LFU correction norm fell from 567 to 410. Thus the ridge state is working
in the intended direction, while this single half-life still admits corrections
that are too large for accurate tracking; half-life and online sample size
remain experimental variables rather than settled constants.

The ridge pilot's CUDA audit passed with bitwise-equal repeated GPU
derivatives and a maximum matched CPU/GPU relative discrepancy of
$4.21\times10^{-15}$, below the $6.04\times10^{-4}$ tolerance. Dense replay
took 6.34 seconds, peak allocated CUDA memory was approximately 447 MiB, and
selected checkpoints occupied approximately 269 MiB including the added ridge
state. Phase 4 is awaiting check-in with the recommendation to retain both raw
and ridge conditions and defer any half-life sweep to the paired experimental
grid.

---

## Phase 5: Results notebook and factor pilot

### Goal

Make the experiment legible before spending substantial compute, then use a small paired pilot to select the main dense grid.

### Scope

1. Create `mnist_experiment/results.ipynb`.
2. The notebook may:
   - discover completed compatible runs;
   - load tidy scalar artifacts;
   - validate schema and configuration compatibility;
   - align trajectories by $p$;
   - show individual and mean trajectories;
   - calculate paired differences and lightweight uncertainty intervals;
   - summarize completed replica counts.
3. The notebook must not:
   - import training entry points as a side effect;
   - download data;
   - calculate gradients, HVPs, eigendecompositions, or reference Fishers;
   - mutate or repair runs.
4. Include an **Assumption checks** section sourced from the completed Phase 3
   artifacts, without recomputing scores or derivatives. It must show:
   - nested reference-Fisher convergence at $p\in\{0.25,0.5,0.75\}$;
   - full-LFU, Amari--Chentsov-only, and residual-only stencil errors across
     tested directions and finite-difference scales;
   - the weighted stencil targeting $(C+R):v$ and the unweighted stencil
     targeting $R:v$;
   - importance-weight means and effective sample sizes;
   - minimum-eigenvalue and PSD diagnostics;
   - reference-Fisher and LFU timing plus peak GPU memory;
   - the selected reference sample sizes and finite-difference scale, including
     sensitivity values;
   - the CUDA adaptive-average-pooling backward warning and the Phase 4
     repeatability-audit result.
   The notebook must select the final compatible Phase 3 runs explicitly and
   must not silently combine them with provisional or exploratory runs.
5. Treat artifact compatibility explicitly through strict, read-only loaders:
   - load Phase 3 artifacts only through their expected assumption-check
     schemas;
   - recognize Phase 4 metric schema v1 as a legacy four-condition run;
   - require Phase 4 metric schema v2 for ridge comparisons and the principal
     Phase 5 pilot;
   - normalize legacy scalar rows only in memory, marking unavailable ridge
     fields and conditions explicitly;
   - reject unknown schemas and incompatible parameter layouts or experimental
     designs with a clear error;
   - never rewrite, migrate, repair, or otherwise mutate a completed run.
6. Prevent duplicated evidence when immutable runs overlap. If schema-v1 and
   schema-v2 runs share the same replica, trajectory hash, and experimental
   design, schema v2 supersedes schema v1 in aggregate analysis. Retain the
   schema-v1 run in provenance displays as a reproducibility check, but never
   count the pair as independent replicas.
7. Run a small paired dense pilot over selected values of:
   - `num_p_steps`;
   - `samples_per_step`;
   - EMA gain or forgetting factor;
   - directional-ridge half-life.
   Hold the ridge amplitude floor and coherence threshold fixed for this pilot.
   Evaluate a small axial half-life set rather than crossing it with the full
   factor grid.
8. Include both factorial and approximately compute-matched comparisons.
9. Treat the replica as the statistical unit. Trajectory points are repeated
   measurements within a replica, not independent samples. Single-replica
   cells may be visualized and used to validate the pipeline, but must not
   report inferential uncertainty as though it were replicated evidence.
10. Estimate the compute and storage cost of adding one replica under each candidate configuration.

### Verification gate

- The notebook executes quickly from completed smoke/pilot artifacts.
- Incomplete or incompatible runs fail clearly.
- Legacy and ridge-capable runs are visibly distinguished, and an overlapping
  schema-v1/v2 pair contributes at most one replica to an aggregate.
- Every plotted average reports contributing replica count.
- Paired differences are calculated by replica, not by treating trajectory points as independent replicas.
- Assumption-check figures reproduce the stored Phase 3 metrics and make the
  numerical validity and CUDA limitation visible without running model
  computations.
- The pilot reveals whether the chosen step/sample ranges produce distinguishable tracking regimes.
- Ridge half-life sensitivity is measured without launching a large Cartesian
  sweep.

### Check-in decisions

- Select the principal dense `num_p_steps` and `samples_per_step` grid.
- Select EMA gains and reference checkpoint cadence.
- Decide how many initial replicas justify proceeding to EWC coupling.
- Lock the EWC proposal optimizer, inner-step count, and regularization convention for Phase 6.

### Pilot design

Use one replica in each cell to validate the analysis pipeline and identify
tracking regimes. The accepted Phase 4 ridge pilot is the center cell
$(K,m,\alpha,h)=(21,128,0.25,8)$.

- Run a $2\times2$ step/sample factorial with
  $K\in\{9,21\}$ and $m\in\{64,128\}$ at $\alpha=0.25$ and $h=8$.
- Treat $(K,m)=(9,128)$ and $(21,64)$ as the approximately compute-matched
  comparison. Their online observation budgets are 1,152 and 1,344.
- Use an axial EMA-gain set $\alpha\in\{0.10,0.25,0.50\}$ at
  $(K,m,h)=(21,128,8)$.
- Use an axial ridge half-life set $h\in\{4,8,16\}$ at
  $(K,m,\alpha)=(21,128,0.25)$.
- Hold $\varepsilon_a=10^{-6}$ and the coherence threshold at 0.75.
- Preserve approximately quarter-trajectory oracle refreshes: cadence 2 for
  $K=9$ and cadence 5 for $K=21$.

This is eight cells total, including the completed center cell, and requires
seven new immutable runs. It is intentionally descriptive: no cell-level
uncertainty interval is reported until additional complete replica trajectories
are accumulated.

### Implementation record

Phase 5 reached its check-in on 2026-07-30. The implementation provides:

- strict read-only Phase 3 and Phase 4 artifact loaders;
- structural parameter-layout, manifest, configuration-hash, trajectory-hash,
  schema, method-set, and grid validation;
- condition-level schema precedence, so legacy schema-v1 runs and repeated raw
  controls from half-life replays cannot inflate replica counts;
- an explicit accepted-run selector that includes the Phase 4 ridge baseline
  and `mnist_lfu_phase5_` experiments while leaving failed-regime and smoke
  runs visible only in provenance;
- an artifact-only notebook with reference convergence, stencil, importance
  weight, PSD, timing, CUDA, trajectory, paired-difference, factor, uncertainty,
  and cost views;
- a source validator that rejects training, data-download, derivative, Fisher,
  and direct tensor-loading operations from notebook cells.

The notebook loads the two explicitly accepted Phase 3 runs covering
$p\in\{0.25,0.5,0.75\}$ and eight principal tracking runs in approximately
3.1 seconds. It never loads the large Fisher checkpoint files. Every current
factor cell contains one replica, so its 95% interval fields are deliberately
unavailable rather than treating trajectory points as replicates.

All seven new immutable GPU runs completed and passed the CUDA audit. The
factorial pilot produced these interior mean relative errors:

| $K$ | $m$ | EMA | full LFU | ridge full, $h=8$ | periodic fresh |
|---:|---:|---:|---:|---:|---:|
| 9 | 64 | 0.528 | 0.793 | 0.470 | 0.183 |
| 9 | 128 | 0.466 | 0.896 | 0.375 | 0.154 |
| 21 | 64 | 0.297 | 1.770 | 1.227 | 0.185 |
| 21 | 128 | 0.283 | 1.661 | 1.163 | 0.176 |

On the approximately compute-matched cells, ridge-full error was 0.375 for
$(K,m)=(9,128)$ and 1.227 for $(21,64)$. Thus repeated correction accumulation,
not merely online observation budget, is a material factor in this replica.
At $K=9$, ridge smoothing modestly outperformed EMA; at $K=21$, it remained
substantially worse. Raw AC and full LFU were worse than EMA in every cell,
while the residual term consistently gave a small improvement over AC-only.

At $(K,m,h)=(21,128,8)$, increasing the direct-Fisher gain from 0.10 to 0.25
to 0.50 reduced EMA error from 0.547 to 0.283 to 0.145 and ridge-full error
from 1.920 to 1.163 to 0.548. At $(K,m,\alpha)=(21,128,0.25)$, increasing ridge
half-life from 4 to 8 to 16 reduced ridge-full error from 1.325 to 1.163 to
0.971 and mean correction norm from 425 to 381 to 327. Both factors improved
through the upper tested boundary, so the pilot identifies a direction but
does not establish an interior optimum.

The accepted center and treatment-axis runs shared an exact trajectory hash.
The $K/m$ cells used fresh deterministic CPU initializations and distinct stream
bundles; all fitted models had the exact same model-state hash as the center.
Directional coherence remained high, no no-motion cases occurred, and each
cell had only one or two resets. Maximum relative PSD-projection distances were
below $10^{-5}$.

Observed trajectory-plus-replay time was approximately 8.8--13.8 seconds per
run after reference caching. Peak allocated GPU memory was approximately
298 MiB for $K=9$ and 447 MiB for $K=21$. Current dense checkpoint policy uses
approximately 268--269 MiB per run; the eight principal runs occupy about
2.10 GiB before shared reference-cache storage.

Phase 5 is awaiting check-in. Before locking Phase 6, decide whether to:

- replicate the four $K/m$ cells or concentrate replicas on the promising
  $(9,128)$ cell and the $(21,64)$ compute-matched comparator;
- extend the gain and half-life axes once because their best values occurred at
  the tested boundaries;
- include substantially smaller `samples_per_step` values to represent the
  intended one-to-few-observation deployment regime;
- retain the validated learning proposal settings: SGD learning rate 0.001,
  three inner steps, EWC strength 1, and a start-of-step anchor.

---

## Phase 6: Dense EWC-coupled experiment

### Goal

Determine whether Fisher-tracking differences change continual-learning behavior when each estimator influences future updates.

### Scope

1. Implement the EWC-regularized estimate using the active dense Fisher and
   the odds-form coefficient

   $$
   \lambda_t=\frac{1-\pi_t}{\pi_t}.
   $$

   Accept the optimizer's realized displacement directly; do not multiply it
   by $\pi_t$ afterward.
2. Begin with a fixed interior adaptation weight
   $\pi_t=\pi_0\in(0,\pi_{\max}]$, common to every estimator condition. This
   keeps the EWC penalty active while isolating Fisher tracking from the
   adaptive controller studied in Phase 8.
3. Treat `optimizer.ewc_strength` as an optional dimensionless sensitivity
   multiplier on the mixture-derived odds:

   $$
   \lambda_t
   =
   \texttt{ewc\_strength}\frac{1-\pi_t}{\pi_t}.
   $$

   Principal Phase 6 runs use `ewc_strength = 1`.
4. Clone the shared initialization and paired observation stream into each estimator condition.
5. Allow each condition's Fisher estimate to alter its optimization result and future parameter path.
6. Preserve shared non-treatment settings while explicitly recording path divergence.
7. Estimate the high-sample reference Fisher at each condition's own configured checkpoints.
8. Record:
   - $p_t$, $\pi_t$, $\lambda_t$, and the objective normalization;
   - data loss and EWC penalty;
   - optimizer displacement norm, with proposal and accepted update defined as
     the same realized displacement in this phase;
   - Fisher-weighted drift;
   - old-digit, digit-9, and balanced metrics;
   - Fisher tracking and projection diagnostics;
   - runtime and memory.
9. Add a smoke run covering all dense estimator conditions.
10. Create `mnist_experiment/coupled_results.ipynb` for Phase 6 review:
   - show each condition's divergent parameter path;
   - compare retention, adaptation, and balanced performance;
   - show data-loss and EWC-penalty trajectories;
   - compare Fisher tracking only against references evaluated at that
     condition's own parameter checkpoint;
   - show proposal norms, accepted displacement norms, and Fisher-weighted
     drift;
   - report paired per-replica learning outcomes without treating checkpoints
     as independent samples.
   The notebook must use the shared strict artifact loaders and remain
   visualization and lightweight statistical analysis only.

### Tests

- identical Fisher inputs yield identical EWC proposals;
- $\pi_t\in(0,\pi_{\max}]$ and
  $\lambda_t=\texttt{ewc\_strength}(1-\pi_t)/\pi_t$;
- changing only the Fisher condition can change the path but not the paired data stream;
- each reference Fisher is keyed to the correct condition checkpoint;
- EWC penalties are nonnegative after projection;
- proposal and realized displacement artifacts agree;
- no post-optimization $\pi_t$ scaling is applied;
- the following LFU uses the realized displacement.

### Verification gate

- All dense conditions complete a small coupled trajectory.
- Divergent paths are evaluated against references at their own parameters.
- Retention and adaptation metrics are reproducible under paired seeds.
- No conclusion depends solely on projected matrices without raw instability diagnostics.
- `coupled_results.ipynb` executes quickly from completed artifacts, makes path
  divergence explicit, and performs no training, derivative, or reference
  computation.

### Check-in decisions

- Decide whether full LFU provides enough tracking or learning signal to justify structured scaling.
- Select the dense conditions and factor settings carried into Phase 7.
- Revise the EWC proposal only if the coupled pilot exposes a documented defect.

### Implementation and pilot record

Phase 6 implemented:

- `src.ewc.mixture_ewc_strength`, which calculates and records
  $\gamma(1-\pi_t)/\pi_t$;
- mixture-weighted EWC proposals that accept the optimizer solution directly
  and record that no post-optimization scaling occurred;
- `src.coupled_trajectory.DenseFisherTracker`, which preserves the validated
  Phase 4 update mathematics while giving each condition its own Fisher state;
- `mnist_experiment/run_coupled.py`, with six paired dense conditions,
  own-path reference Fishers, lagged realized directions, classifier metrics,
  path-divergence diagnostics, immutable trajectories, and matrix checkpoints;
- strict Phase 6 artifact loaders and summaries in `src.results_analysis`;
- `mnist_experiment/coupled_results.ipynb`, which selects the accepted pilot,
  performs no training or derivative work, and fails on incompatible artifacts;
- `phase6_smoke.json`, `phase6_pilot.json`, and
  `phase6_convergent_pilot.json`.

All Phase 6 runs use fixed $\pi_0=0.5$, $\pi_{\max}=0.95$, and
`ewc_strength = 1`, hence an effective odds coefficient of one. The smoke run
completed all six conditions over three steps and verified shared
initialization, shared observation identities, own-path references, exact
lagged-displacement scheduling, and the absence of post-optimization
actuation. Its final maximum pairwise parameter distance was
$1.13\times10^{-7}$; this tiny run is a software check, not scientific
evidence.

The first $K=9,m=128$ GPU pilot retained the Phase 5 proposal settings of three
SGD steps at learning rate $10^{-3}$. It completed, but every condition had
zero digit-9 accuracy and weak likelihood adaptation. This documented
under-actuation justified revising the original-process inner solve. Increasing
the learning rate was not safe: exploratory settings
$(10,10^{-2})$, $(5,5\times10^{-3})$, and $(5,3\times10^{-3})$ produced
nonfinite derivatives in at least one condition. The two contextualized
failures occurred for ridge AC-only at $p=1$ and raw AC-only at $p=1$.
These failed attempts did not produce completed artifacts.

The accepted convergent pilot instead used 20 small SGD steps at learning rate
$10^{-3}$. All six conditions completed with high-sample
$N_{\mathrm{ref}}=32{,}768$ references at their own checkpoints. Observed
interior mean Fisher errors were:

| Condition | Mean relative Fisher error | Final digit-9 NLL | Digit-9 NLL reduction | Final non-nine accuracy |
|---|---:|---:|---:|---:|
| EMA | 0.462 | 16.729 | 13.447 | 0.885 |
| AC only | 0.898 | 24.627 | 5.548 | 0.886 |
| Full LFU | 0.879 | 24.607 | 5.568 | 0.886 |
| Periodic fresh | 0.155 | 21.134 | 9.041 | 0.885 |
| Ridge AC only | 0.271 | 19.911 | 10.264 | 0.888 |
| Ridge full LFU | 0.271 | 19.906 | 10.269 | 0.888 |

The final maximum pairwise parameter distance was 0.252, so the coupled paths
were materially distinct. Runtime was 110.4 seconds and peak allocated GPU
memory was approximately 636 MiB. Maximum relative PSD-projection distance was
$6.88\times10^{-5}$ for raw full LFU, $1.43\times10^{-5}$ for AC-only,
$6.78\times10^{-9}$ for ridge full LFU, and at numerical noise level for EMA,
periodic fresh, and ridge AC-only.

Digit-9 accuracy remained zero because fitting at $p=0$ drives the unseen class
toward a softmax boundary; nevertheless, digit-9 NLL decreased substantially
and provides a continuous adaptation metric. This endpoint behavior must remain
visible in the notebook and should not be mistaken for absence of movement.
The accepted pilot has one replica and the CUDA adaptive-average-pooling
backward remains nondeterministic, so all method comparisons are descriptive.

Verification completed with 88 unit tests and an artifact-only notebook
execution time of approximately 1.7 seconds. Phase 6 is awaiting check-in.
Before Phase 7, decide whether to carry:

- EMA as the primary low-compute control;
- periodic fresh as the expensive reference-like learning condition;
- ridge full LFU as the principal LFU treatment;
- raw full LFU only as an instability/ablation condition, given its poor
  tracking and weak adaptation in this replica.

---

## Phase 7: Diagonal and low-rank-plus-diagonal representations

### Goal

Measure how much Fisher and LFU information survives practical representation constraints.

### Scope

1. Define a small representation interface for:
   - matrix-vector products;
   - diagonal extraction;
   - EWC quadratic forms;
   - damped solves or inverse traces where supported;
   - serialization;
   - diagnostics.
2. Implement the diagonal representation:
   - update only the diagonals of $Z$, $\widehat\Delta_C$, and $\widehat\Delta_R$;
   - project by clipping negative entries;
   - test against the diagonal of the dense update.
3. Implement a wrapper around the copied `src/lanczos.py`:
   - deterministic randomized initialization;
   - symmetric matrix-vector operator input;
   - device and dtype control;
   - requested and realized rank checks;
   - residual-diagonal construction;
   - stable artifact serialization.
4. Represent the approximation as

   $$
   \widehat{\mathcal I}=AA^T+\operatorname{diag}(d),
   \qquad d_i\geq0.
   $$

5. Apply Lanczos to the complete updated operator, retaining positive Ritz components and a clipped residual diagonal.
6. Evaluate a targeted rank grid selected from dense results. Include rank zero as the diagonal-only boundary.
   The initial grid is $r\in\{0,4,8,16,32,64\}$ for the 512-parameter
   canonical network. One immutable validation run evaluates the complete grid
   on one paired fixed trajectory so randomized CUDA trajectory differences
   cannot masquerade as rank effects.
7. Compare against dense projected checkpoints using:
   - relative Frobenius error;
   - fixed-probe matrix-vector error;
   - EWC quadratic-form error;
   - leading eigenspace error;
   - memory and runtime.
8. Create `mnist_experiment/representation_results.ipynb` for Phase 7 review:
   - show diagonal and low-rank-plus-diagonal accuracy by retained rank;
   - compare fixed-probe matrix-vector and EWC quadratic-form errors;
   - show leading-eigenspace accuracy;
   - display memory, runtime, and artifact-size frontiers;
   - distinguish fixed-trajectory validation from any surviving coupled runs;
   - identify Pareto-efficient representation/rank choices without hiding
     failed or numerically unstable settings.
   The notebook must use precomputed scalar metrics and shared artifact
   validation; it must never reconstruct large matrices merely for plotting.

### Tests

- diagonal results equal the diagonal of the corresponding dense operation;
- low-rank-plus-diagonal matrix-vector products match explicit reconstruction;
- rank-zero behavior matches the documented diagonal boundary;
- deterministic seeds reproduce Lanczos outputs within sign/subspace conventions;
- invalid rank and breakdown cases fail clearly;
- the external Lanczos source remains unchanged;
- serialization preserves factors and parameter ordering.
- the copied `src/lanczos.py` has SHA-256
  `643bb7562ad7ad2d087229414d7adf114f575b342f462be619f90be920691634`.

### Verification gate

- Diagonal and each selected rank complete fixed-trajectory smoke runs.
- At least one low-rank setting improves materially over diagonal on a dense-reference metric, or evidence clearly shows that it does not.
- Approximation quality, compute, and storage form an interpretable rank curve.
- Coupled runs are attempted only for ranks that survive fixed-trajectory validation.
- `representation_results.ipynb` executes quickly from stored metrics and
  exposes the rank/accuracy/compute frontier without model computation.

### Check-in decisions

- Select ranks for coupled and controller experiments.
- Decide whether the copied Lanczos implementation needs a corrective wrapper behavior or whether a separately versioned replacement is warranted.
- Decide which representation offers the best accuracy/compute frontier.

### Implementation record

Phase 7 reached its check-in on 2026-07-30. The implementation provides:

- dense, diagonal, and low-rank-plus-diagonal Fisher objects with matrix-vector
  products, quadratic forms, damped solves, inverse traces, serialization, and
  explicit persistent-storage accounting;
- schema-v6 rank grids and immutable fixed/coupled runners;
- a deterministic wrapper around the unchanged copied Lanczos source, pinned
  to SHA-256
  `643bb7562ad7ad2d087229414d7adf114f575b342f462be619f90be920691634`;
- recursive diagonal and low-rank-plus-diagonal ridge-full LFU trackers;
- structured EWC penalties that use the factorization directly rather than
  reconstructing a dense matrix during optimization;
- strict read-only Phase 7 loaders that distinguish artifact completion from
  numerical acceptability; and
- `representation_results.ipynb`, which loads only scalar artifacts and
  displays fixed-path and coupled-path evidence separately.

The immutable fixed-path pilot
`mnist_lfu_phase7_rank_pilot__replica-0000__14916102ebf92cc3`
completed the requested ranks $\{0,4,8,16,32,64\}$ on one shared nine-step
trajectory. Mean dense ridge-full target errors, excluding the initial step,
were 0.978 for diagonal, 0.198 for rank 4, and 0.104 for rank 8. Rank 8 also
had mean fixed-probe matrix-vector error 0.131, mean EWC quadratic error 0.098,
and mean leading-eigenvector alignment 0.9999. Its persistent factor and
diagonal occupied 36 KiB, versus 2,048 KiB for one dense float64 Fisher.

Higher requested ranks were not monotone improvements. Mean dense-target error
rose to 5.07, 105, and 87,686 for ranks 16, 32, and 64. Their clipped
residual-diagonal mismatch and exploding retained Ritz values show numerical
loss of control in the copied unreorthogonalized recurrence. The analysis
therefore labels these finite artifacts `numerically_unstable`. It does not
rewrite or discard them. The copied implementation remains unchanged and
usable through the guarded wrapper at rank 8, but any future higher-rank study
should use a separately versioned reorthogonalized implementation rather than
quietly changing the provenance-pinned source.

Only rank 8 survived into the immutable coupled run
`mnist_lfu_phase7_coupled_rank8__replica-0000__b0f9fdc792134eb9`.
Its mean path-specific Fisher error was 0.255, close to dense ridge-full at
0.246, and its final Fisher error was 0.0738 versus 0.0722. The final rank-8
parameter vector was only 0.00490 from the dense vector. Their final NLL,
digit-9 NLL, non-nine NLL, balanced accuracy, and distance from initialization
were likewise nearly equal. Rank 8 therefore preserved both the auxiliary
Fisher process and the original learning process in this replica.

The diagonal condition behaved qualitatively differently. Its mean Fisher
error was 2.07, final error was 7.49, and its final parameter vector was 0.792
from dense. It adapted much more aggressively to digit 9 while forgetting
non-nine performance, reducing balanced accuracy to 0.624 versus approximately
0.798 for dense and rank 8. Diagonal remains a useful low-memory boundary and
ablation, not the recommended controller representation.

The fixed sweep took 9.06 seconds and produced 22.5 MiB of validation
artifacts. The three-condition coupled run took 49.7 seconds and produced
103.8 MiB, primarily because scientific validation retains selected dense
references and checkpoints; those artifact sizes are not the persistent
runtime state sizes. Mean Fisher-update times were 0.0566 seconds for dense
and 0.0507 seconds for rank 8 in the coupled pilot.

Verification completed with 105 unit tests. The CPU fixed and coupled smoke
runs completed all selected conditions and immutable reruns were refused. The
artifact-only notebook executes in approximately one second. Phase 7 is
awaiting check-in with the recommendation to carry dense ridge-full as the
validation comparator, rank-8 low-rank-plus-diagonal ridge-full as the
practical representation, and diagonal ridge-full as a boundary ablation into
the controller design discussion.

---

## Phase 8: Optimal-controller experiment

### Goal

Evaluate the adaptation controller after the principal Fisher estimators and representations have been selected.

### Post-Phase-7 paradigm clarification

The accepted Phase 6 and Phase 7 coupled pilots used `fixed_pi = 0.5` in the
mixture-weighted EWC objective and an independent `ema_gain = 0.25` in the
auxiliary Fisher recursion. Their immutable artifacts remain valid
measurements of those explicitly decoupled conditions, but the conditions are
theoretical aberrations under the subsequently accepted $\pi_t$-centric
paradigm. Treat them as historical decoupled baselines only. Do not use their
coupling behavior as evidence for or against the unified controller, and do
not rewrite or discard their artifacts.

Phase 8 must introduce new configuration, metric, and trajectory schema
versions. Each step records one applied $\pi_t$, and that same value must be
consumed by both:

$$
\widehat{\mathcal I}_t
=
(1-\pi_t)
\left(\widehat{\mathcal I}_{t-1}+\widehat\Delta_t^{\mathrm{lag}}\right)
+\pi_t Z_t,
$$

and the mixture-weighted EWC objective with odds
$(1-\pi_t)/\pi_t$. No independent Fisher gain or precision-forgetting
hyperparameter belongs to a principal Phase 8 condition. A small paired
compatibility run should compare the unified fixed-$\pi=0.5$ condition with
the historical decoupled baseline before adaptive-controller conclusions are
drawn.

### Resolved open questions

1. **Asymptotic experiment.** The principal controller uses the fixed-batch
   stratified EWC model with new-batch covariance proportional to
   $\pi_t^2/m_t$. The fixed-total Bernoulli-composition model, whose covariance
   is proportional to $\pi_t/N$, remains a theoretically convenient oracle
   diagnostic and is not substituted for the applied risk.
2. **Limits and noise laws.** The Bernoulli small-noise model has coefficient
   $\sqrt{\varepsilon\pi_t}\,\mathcal I^{-1/2}$. The oracle-recentered
   fixed-batch interpolation has coefficient
   $\sqrt\varepsilon\,\pi_t\mathcal I^{-1/2}$. The finite applied
   tracking-error recursion is primary because it retains random anchor error.
3. **Meaning of $\pi_t$.** In the applied model, $\pi_t$ is the normalized
   new-data weight, adaptation rate, EWC forgetting rate, and direct-Fisher
   blend weight. It is not a claim that the fixed observed batch was generated
   by Bernoulli thinning.
4. **Signal.** The target signal is
   $d\theta_t=\theta_{t+1}^\star-\theta_t^\star$. The plug-in estimates its
   local trend from accepted EWC displacements; the oracle obtains it from a
   high-sample reference-optimum path.
5. **Trend estimator.** A single accepted update is policy dependent, but
   $u_t=d\theta_t+(e_{t+1}-e_t)$. Therefore a vector EMA of accepted updates
   estimates the local trend when tracking error is locally stationary. No
   disposable unregularized small-batch MLE belongs to a principal run.
6. **Effective information size.** Assume covariance calibration and use

   $$
   q_t=(1-\pi_t)^2q_{t-1}+\frac{\pi_t^2}{m_t},
   \qquad N_{\mathrm{eff},t}=q_t^{-1}.
   $$

   Fisher Monte Carlo sample size remains a separate recorded quantity.
7. **Trace estimation.** Estimate
   $T_t=\operatorname{tr}\mathcal I(\theta_t)^{-1}$ without inversion from
   predictable accepted-update residuals. With

   $$
   r_t=u_t-\widehat d_{t\mid t-1},
   \qquad
   a_t=\pi_t^2
   \left(N_{\mathrm{eff},t-1}^{-1}+m_t^{-1}\right),
   $$

   use exponentially weighted moments $V_t\approx\mathbb E\|r_t\|^2$ and
   $A_t\approx\mathbb E a_t$, then set
   $\widehat T_t=V_t/(A_t+\varepsilon)$. This requires local covariance and
   trend stability and is tested rather than assumed silently.
8. **Loss.** Local optimality uses coordinate-local Euclidean parameter MSE.
   The coordinate dependence is accepted and recorded; do not replace it with
   a Fisher-metric controller without a new derivation.
9. **Boundaries.** Principal policies use

   $$
   \pi_{\mathrm{used},t}
   =\min(\pi_{\max},\max(\pi_{\min},\widehat\pi_t^\star)),
   $$

   with $\pi_{\min}>0$. The lower bound maintains excitation, finite EWC
   odds, and recovery after quiet periods. A hard freeze is a separate
   explicit action, and `uncontrolled` is an explicit $\pi=1$ boundary.
10. **Oracle knowledge.** `optimal_oracle` uses a high-sample, warm-started
    reference-optimum path for $d\theta_t$ and a high-sample reference Fisher
    for covariance terms. The Bernoulli oracle formula is also recorded as a
    theory diagnostic, but it is not presented as the deployable controller.
11. **Predictability.** Calculate $\pi_t$ only from summaries available
    through step $t-1$. Apply it to batch $t$, then update trend and trace
    states after accepting $u_t$. Same-batch controller estimation is excluded
    from principal policies.
12. **Information horizon.** The same $\pi_t$ controls adaptation, direct
    Fisher blending, and EWC forgetting. Track the realized products of
    $1-\pi_t$ and the $q_t$ recursion; do not introduce an independent Fisher
    gain or precision-forgetting factor.
13. **Actuation.** Optimize the mixture-weighted EWC objective with odds
    $(1-\pi_t)/\pi_t$ and accept its optimizer displacement directly. Never
    assign $u_t\leftarrow\pi_t\widetilde u_t$ after optimization.

### Applied controller

Let $T_t:=\operatorname{tr}\mathcal I(\theta_t)^{-1}$ and define

$$
\tau_{\mathrm{old},t}=\frac{T_t}{N_{\mathrm{eff},t}},
\qquad
\tau_{\mathrm{new},t}=\frac{T_t}{m_{t+1}}.
$$

The fixed-batch risk and oracle minimizer are

$$
R_{B,t}(\pi)
=
(1-\pi)^2
\left(\|d\theta_t\|^2+\tau_{\mathrm{old},t}\right)
+\pi^2\tau_{\mathrm{new},t},
$$

$$
\pi_{B,t}^\star
=
\frac{
\|d\theta_t\|^2+\tau_{\mathrm{old},t}
}{
\|d\theta_t\|^2+\tau_{\mathrm{old},t}+\tau_{\mathrm{new},t}
}.
$$

The plug-in substitutes the predictable vector trend and scalar trace
estimates, then applies `pi_min` and `pi_max`. The default values are
`pi_min = 0.05`, `pi_max = 0.95`, and `trend_half_life_p = 0.20`. Use targeted
sensitivity sets `pi_min` in `{0.01, 0.05, 0.10}` and
`pi_max` in `{0.80, 0.95, 1.00}`, and `trend_half_life_p` in
`{0.10, 0.20, 0.40}` only after the principal smoke run.

For an environmental increment $\Delta p_t$, use

$$
\gamma_t=1-2^{-\Delta p_t/H_p}.
$$

Initialize $\widehat d_{0\mid-1}=0$, $q_0=N_{\mathrm{init}}^{-1}$, and the
trace moment accumulators at zero. Until one trend half-life of environmental
distance has accumulated, use the bounded zero-trend composition

$$
\pi_{\mathrm{cold},t}
=
\operatorname{clip}_{[\pi_{\min},\pi_{\max}]}
\left(\frac{m_t}{N_{\mathrm{eff},t-1}+m_t}\right).
$$

After accepting $u_t$, update

$$
\widehat d_{t+1\mid t}
=(1-\gamma_t)\widehat d_{t\mid t-1}+\gamma_tu_t,
$$

$$
V_t=(1-\gamma_t)V_{t-1}+\gamma_t\|r_t\|^2,
\qquad
A_t=(1-\gamma_t)A_{t-1}+\gamma_ta_t,
$$

and then $q_t$. Store all pre-update and post-update states so the notebook can
reconstruct every decision without loading a model.

### Implementation scope

1. Before controller code, update `mnist_experiment/AGENTS.md` to replace the
   obsolete independent Fisher-gain/controller contract with this resolved
   design.
2. Introduce configuration schema version 7 and new Phase 8 metric,
   trajectory, controller-state, and oracle-path artifact schema versions.
   Legacy schema versions remain readable only through their existing strict
   loaders and are never upgraded in place.
3. Add controller configuration fields for policy, `fixed_pi`, `pi_min`,
   `pi_max`, `trend_half_life_p`, trace numerical epsilon, and oracle mode.
   Unified policies must not accept `ema_gain` or a precision-forgetting
   factor. Preserve those fields only in legacy decoupled schemas.
4. Implement a small typed controller state containing:
   - the predictable vector trend;
   - $q_t$ and $N_{\mathrm{eff},t}$;
   - $V_t$, $A_t$, and $\widehat T_t$;
   - estimated old and new covariance traces;
   - accumulated environmental distance and warm-up status;
   - previous applied $\pi_t$ and all numerical-guard flags.
5. At each step, enforce this ordering:
   - calculate raw and bounded $\pi_t$ from state through $t-1$;
   - use that exact value in both the Fisher blend and EWC odds;
   - optimize the EWC objective and accept its displacement without scaling;
   - use the accepted displacement in the next LFU and to update controller
     state for step $t+1$.
6. Implement policies:
   - `uncontrolled`, an explicit $\pi=1$ boundary;
   - `fixed_unified`, including the $\pi=0.5$ compatibility run;
   - `optimal_plugin`, the bounded applied controller;
   - `optimal_oracle`, the bounded fixed-batch controller using reference
     displacement and an oracle-trend residual covariance moment;
   - `freeze`, an explicit no-update action outside the ordinary bounds.
   Record the Bernoulli-theory oracle value at each step as a diagnostic, not
   as the principal applied policy.
7. Build one immutable high-sample reference-optimum path per replica and
   $p$ grid. Warm-start each point from the preceding reference optimum to
   select a continuous branch, record convergence diagnostics, and share the
   path across paired controller conditions. Calculate oracle displacements
   from adjacent reference points. Never invert or pseudoinvert a reference
   Fisher for controller covariance.
8. Do not run a disposable unregularized optimization in principal plug-in
   trajectories. Optional paired or high-sample probes may run only at
   configured assumption-check checkpoints and must be stored as separate
   diagnostic artifacts.
9. Add `mnist_experiment/run_controller.py` for immutable Phase 8 runs and
   `mnist_experiment/controller_results.ipynb` for artifact-only analysis.
   Carry only dense ridge-full, rank-8 low-rank-plus-diagonal ridge-full, and
   the minimum boundary ablations selected by Phase 7.
10. Run a small paired compatibility experiment comparing unified fixed
    $\pi=0.5$ with the historical decoupled $\pi=0.5$, `ema_gain=0.25`
    baseline before adaptive-policy pilots. Never merge their schema or
    interpretation.

### Assumption checks

- Compare the plug-in trend with oracle $d\theta_t$ using norm error, cosine
  similarity, turning angle, and one-step prediction error.
- Show path curvature through adjacent oracle-displacement turning angles and
  second differences; compare the three configured trend half-lives.
- Compare $\widehat T_t/N_{\mathrm{eff},t}$ with empirical squared parameter
  error around the reference path across replicas.
- Compare the deployable trace moment based on
  $u_t-\\widehat d_{t\\mid t-1}$ with the oracle-trend moment based on
  $u_t-d\\theta_t$. Their difference measures variance contamination from
  trend estimation. Neither controller estimate may invert or pseudoinvert a
  Fisher matrix.
- Report residual autocorrelation and the stability of
  $\|r_t\|^2/a_t$ over each local half-life. Material autocorrelation or drift
  invalidates the simple moment interpretation.
- Check projected error distributions for approximate centering and LAN-scale
  behavior without claiming exact multivariate Gaussianity.
- Record the local-quadratic diagnostic, PSD projection distance, LFU
  truncation indicators, and whether either $\pi$ bound was active.

### Required artifacts

Every trajectory row must record raw, cold-start, oracle, theory-oracle, and
applied $\pi$ values; applied bounds and activation flags; EWC odds; policy;
warm-up status; $\gamma_t$; $q_t$; $N_{\mathrm{eff},t}$; trend norm; residual
norm; $a_t$; $V_t$; $A_t$; $\widehat T_t$; both covariance-trace estimates;
accepted update norms; Fisher-update diagnostics; and objective metrics.
Store controller vectors and reference-optimum vectors in separate checkpoint
artifacts rather than scalar tables.

### Tests

- all ordinary policies remain in `[pi_min, pi_max]`, while explicit
  `uncontrolled` and `freeze` boundaries return one and zero respectively;
- invalid bounds, half-lives, batch sizes, and initial effective sizes fail
  validation;
- the same recorded $\pi_t$ is consumed by the Fisher blend and EWC odds;
- accepted optimizer displacements are never multiplied by $\pi_t$;
- LFU consumes the realized accepted displacement on the next step;
- the $q_t$, half-life gain, cold-start, trend, residual, and trace-moment
  recursions match hand-computed examples;
- synthetic locally linear calibrated trajectories recover trend and trace
  within statistical tolerance, including variable $\pi_t$;
- near-`pi_min` decisions remain finite and retain prior trace information;
- plugin decisions use only prior controller state and pass a predictability
  audit;
- oracle displacements equal adjacent reference-optimum differences;
- unified schema-7 runs reject `ema_gain` and immutable collisions;
- legacy decoupled artifacts remain loadable and cannot be mislabeled as
  unified conditions;
- controller artifacts reproduce every decision exactly.

### Verification gate

- The reference-optimum path converges sufficiently at every selected stencil
  and remains on a continuous warm-started branch.
- Unified fixed-$\pi=0.5$ and historical decoupled compatibility results are
  visible and separately labeled.
- All selected policies complete paired CPU smoke and GPU pilot trajectories.
- Assumption-check plots expose trend lag, covariance calibration, trace
  error, residual dependence, path curvature, and bound activation.
- The default half-life and bounds are not selected from a single favorable
  replica, and targeted sensitivity includes both active and inactive bounds.
- The oracle condition separates controller quality from plug-in estimation
  error.
- `controller_results.ipynb` executes quickly from immutable artifacts and
  performs no training, HVP, Fisher construction, or model loading.

### Check-in decisions

- Decide whether the applied plug-in controller improves tracking and
  retention enough to justify its state and estimation assumptions.
- Decide whether trend or trace estimation is the dominant gap from
  `optimal_oracle`.
- Select any `pi_min`, `pi_max`, and trend-half-life settings worthy of
  replication.
- Confirm that conclusions survive the fixed-$\pi$ compatibility comparison
  and are not artifacts of PSD projection, warm-up, or oracle-path quality.

### Implementation record: pre-GPU check-in

Phase 8 reached its implementation and CPU-smoke check-in on 2026-08-01. The
implementation now provides:

- strict configuration schema v7, which rejects `ema_gain`, controller
  damping, and legacy policy names while preserving schema-v4 through
  schema-v6 hashes and loaders;
- a predictable controller state with vector trend, effective-size recursion,
  scalar residual and scale moments, cold start, both ordinary bounds, and
  explicit uncontrolled and freeze boundaries;
- fixed-batch plug-in and reference-oracle policies plus the fixed-total
  Bernoulli oracle as a recorded theory diagnostic;
- dynamic dense, diagonal, and low-rank-plus-diagonal Fisher blending using
  the exact same recorded $\pi_t$ as the EWC objective;
- a warm-started high-sample reference-optimum path with adjacent oracle
  displacements and optimization diagnostics;
- immutable controller trajectory, controller-state, checkpoint, and oracle
  artifacts through `mnist_experiment/run_controller.py`;
- strict read-only Phase 8 loaders and
  `mnist_experiment/controller_results.ipynb`, which executes without loading
  matrix checkpoints or invoking training code.

Two three-point CPU smoke runs completed: `optimal_plugin` and unified fixed
$\pi=0.5`. Both used the same existing treatment-independent replica bundle,
the same observations, and the same oracle path hash. Every stored row passed
the exact shared-$\pi$ assertion, accepted EWC optimizer displacements without
post-scaling, and a repeated invocation was refused after `COMPLETED`. The
notebook also displays the historical schema-v6 $\pi=0.5$, `ema_gain=0.25`
condition under an explicit decoupled label.

These tiny runs are software checks only. Their reference-optimum fits used
eight observations and retained final minibatch gradient norms from 0.44 to
1.02, so the oracle convergence gate failed visibly. The plug-in policy stayed
at `pi_min=0.05`, and its inversion-free trace estimates differed almost
completely from the now-retired pseudoinverse diagnostic. A calibrated synthetic
variable-$\pi$ process recovered the trace within 20%. Its oracle-trend
comparator showed that most of the approximately 16% upward error in that
single realization was finite-window moment variability rather than trend
error. This is consistent with the planned trend-timescale and residual-
dependence assumption checks; it is not evidence for selecting a controller
setting.

Do not mark Phase 8 complete or launch the substantive GPU pilot before the
check-in. The next run must use a converged reference-optimum path and then
exercise `optimal_oracle` alongside the plug-in policy. Uncontrolled and freeze
remain boundary diagnostics rather than principal candidates. Metric schema v3
removes the pseudoinverse diagnostic and instead maintains a second
inversion-free residual moment using the oracle trend.

The substantive run uses `num_p_steps = 100` and must expose progress through
terminal-safe `tqdm` bars and step-level completion logs. Before launch, add
adaptive oracle convergence with a minimum independent chunk count, a hard
maximum sample or optimization budget, and a configurable six-sigma relative
radius. For Fisher means, estimate variance in Frobenius geometry rather than
using only the scalar variance of matrix norms. Apply the analogous Hilbert-
space confidence-radius construction to oracle score means and reference-path
fit diagnostics. The stopping tolerances remain a pre-run scientific decision.

The pre-run decision is now fixed. Configuration schema v8 and metric schema
v4 use `convergence_sigma = 6`, `convergence_relative_epsilon = 0.01`, an
explicit `convergence_absolute_epsilon = 1e-8`, at least eight independent
chunks or fits, and hard ceilings of 32,768 Fisher scores and 32 independent
reference fits. For tensor observations $Y_b$, maintain the Hilbert-space
Welford moment

$$
S_B=\sum_{b=1}^B\|Y_b-\overline Y_B\|_H^2
$$

and stop when

$$
6\sqrt{S_B/[B(B-1)]}
\leq
10^{-8}+0.01\|\overline Y_B\|_H.
$$

Use Frobenius geometry for independent Fisher chunk means and Euclidean
geometry for independent reference-fit endpoints and adjacent displacement
vectors. The initial reference point is checked using endpoint dispersion;
subsequent points are checked using the displacement radius directly. Warm
start every fit trajectory along its own previous endpoint, average the final
independent trajectories to form the oracle path, and preserve every fit's
validation-score confidence ball. A maximum-budget result remains a valid
artifact but fails the scientific convergence gate visibly.

Repeated MNIST indices do not by themselves make independently generated draw
positions statistically coupled; they are expected when sampling with
replacement from a finite empirical distribution. Nevertheless, metric schema
v4 records within-plan duplicate fractions, adjacent Fisher-chunk index
overlap, training/validation unique-index overlap, and lag-one Hilbert
correlations so finite-dataset dependence is inspectable rather than assumed
away. Controller residual checks subtract the predictable trend before
reporting vector and squared-norm lag-one correlations and the coefficient of
variation of $\|r_t\|^2/a_t$ over each information half-life.

No Phase 8 covariance diagnostic may invert or pseudoinvert a Fisher. The
deployable and oracle-trend covariance traces continue to come from accepted-
displacement residual moments. The paired substantive configurations are
`mnist_experiment/configs/phase8_gpu_convergence.json` for `optimal_plugin` and
`mnist_experiment/configs/phase8_gpu_oracle_convergence.json` for
`optimal_oracle`. They share one replica-design hash and use 100 $p$ points,
batch size 128, rank eight, and terminal-safe progress displays. They are ready
for the user-managed `tmux` run but have not been launched by the implementation
phase.

The schema-v8 CPU smoke
`mnist_lfu_phase8_convergence_smoke__replica-0000__e4199af979c92b67`
completed all three representations. As intended, its two independent fits and
16-score Fisher ceiling reached their hard budgets and failed the scientific
confidence gates visibly. The artifact records no Fisher inverse, all
Frobenius and displacement radii, overlap diagnostics, and trend-removed
residual dependence summaries. The artifact-only controller notebook loads
the mixed schema-v2 through schema-v4 run directory and executes in about two
seconds. The complete fast unit suite contains 134 passing tests. This smoke
validates software and artifact behavior only; it does not alter the pending
substantive-run gate.

The first substantive plugin launch exposed a separate optimizer-stability
guard requirement. At step 29, the dense projected Fisher remained finite and
PSD but its leading eigenvalue reached approximately 1,139. With
`pi_min = 0.05`, the EWC odds were 19, so fixed-step SGD at 0.001 had quadratic
stability product approximately 21.6, far beyond the explicit-Euler bound of
two. The penalty diverged before the controller's finite-displacement guard
stopped the run. EWC proposals must therefore use transactional monotone
objective backtracking, record rejected steps, maximum reductions, and the
minimum accepted learning rate, and restore the anchor and optimizer state if
no finite descent step exists. This changes only the numerical solution of the
already specified mixture-weighted objective; it does not change `pi`, the EWC
estimand, or post-proposal actuation. A completed adaptive reference-optimum
path must also be checkpointed immediately in the incomplete run and reloaded
under `--resume`, so a later controller failure cannot discard that expensive
stage.

The schema-v3 CPU smoke suite subsequently completed all five runner policies:
`optimal_plugin`, `optimal_oracle`, `fixed_unified`, `uncontrolled`, and
`freeze`. Every condition used the same oracle-path hash. The freeze run first
failed safely because infinite EWC odds are not strict JSON; resumption then
completed after the explicit freeze action was changed to record null odds and
`hard_freeze = true`. All schema-v3 metrics record `fisher_inverse_used` as
false, and the full unit suite passes. The remaining pre-compute gate is the
adaptive oracle convergence contract and its configured tolerances, not a
pending controller branch.

---

## Phase 9: Replication, hardening, and handoff

### Goal

Make it safe and straightforward for the user to add compute over time and determine when the experiment has sufficient statistical power.

### Scope

1. Freeze versioned configuration and artifact schemas for the first full experiment.
2. Provide configuration generation for the selected, reduced condition grid.
3. Ensure each replica bundle:
   - spans $p=0$ through $p=1$;
   - pairs every intended condition;
   - writes independent immutable run directories;
   - can be added without modifying previous runs.
4. Harden interruption, incomplete-run, disk-space, and duplicate-run handling.
5. Complete the results notebook with:
   - mean trajectories;
   - uncertainty bands;
   - paired method differences;
   - factor summaries for `num_p_steps` and `samples_per_step`;
   - rank/compute frontiers;
   - controller summaries;
   - completed-replica and uncertainty diagnostics.
   Keep `results.ipynb` as the master cross-phase synthesis and replication
   dashboard. It should summarize and link the accepted conclusions from:
   - `coupled_results.ipynb`;
   - `representation_results.ipynb`;
   - `controller_results.ipynb`.
   Do not duplicate their detailed diagnostic views in the master notebook.
6. Add documented commands for:
   - one smoke replica;
   - one selected GPU replica bundle;
   - adding more replicas;
   - rebuilding the notebook from existing artifacts.
7. Run the full unit suite and the tiny integration trajectory. Long production runs remain user-controlled.

### Verification gate

- Interrupted and duplicate-run scenarios preserve completed artifacts.
- Adding a replica changes aggregate uncertainty without changing earlier results.
- The notebook performs no expensive model computation.
- All four notebooks use shared loaders, explicit accepted-run selectors, and
  the common artifact-only notebook validator.
- Every reported aggregate links back to contributing immutable runs.
- A fresh process can reproduce a smoke run and load its results using only documented commands.
- Remaining scientific limitations are documented without being hidden by implementation details.

### Final check-in

- Approve the reduced production condition grid.
- Decide the initial replica count and compute budget.
- Decide what uncertainty or effect-size evidence will trigger additional replicas.
- Identify which findings, if any, are strong enough to motivate the robotics demonstration.

## Explicitly deferred work

The following are outside this plan unless a phase check-in adds them:

- alternative network architectures or external image downsampling;
- KFAC, block-diagonal, or other Fisher representations;
- modifying the external Lanczos source;
- robotics or reinforcement-learning demonstrations;
- automatic cloud or cluster orchestration;
- training or reference-Fisher computation inside notebooks;
- an exhaustive Cartesian parameter sweep;
- claims of optimality beyond the measured experimental conditions.
