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

The implementation must support immutable paired replicas, dense reference calculations, diagonal and low-rank-plus-diagonal approximations, and a final controller experiment using

$$
u_t=\pi_t\widetilde u_t.
$$

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
| 4 | Dense fixed-trajectory experiment | Awaiting check-in |
| 5 | Results notebook and factor pilot | Pending |
| 6 | Dense EWC-coupled experiment | Pending |
| 7 | Diagonal and low-rank-plus-diagonal representations | Pending |
| 8 | Optimal-controller experiment | Pending |
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

---

## Phase 0: Foundations and run contracts

### Goal

Create the smallest stable software and artifact foundation needed by every later phase.

### Scope

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

Initialization and temporary validation trajectories execute on CPU with
deterministic PyTorch algorithms. GPU score and HVP measurements use
deterministic warning mode because PyTorch does not provide a deterministic
CUDA backward implementation for adaptive average pooling; their completed
matrix artifacts are content-addressed and immutable.

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
- the smoke run creates a valid immutable artifact and refuses overwrite.

### Verification gate

- One complete smoke replica runs from $p=0$ to $p=1$.
- All four methods produce aligned, loadable trajectories.
- Reference comparisons use the same parameter checkpoints.
- No estimator has unexplained NaNs, asymmetry, or unrecorded projection.
- CUDA repeatability error is quantified and judged against explicit scientific
  tolerances before the GPU pilot is accepted.
- Runtime and storage estimates are available for a realistic pilot.

### Check-in decisions

- Confirm the trajectory driver is scientifically neutral enough.
- Confirm the EMA parameterization and periodic-fresh cadence.
- Decide whether dense PSD projection occurs every step or at a justified configured cadence based on measured cost.
- Approve the pilot grid for `num_p_steps` and `samples_per_step`.

### Implementation record

Phase 4 reached its check-in on 2026-07-28. The implementation provides:

- an oracle-assisted EWC trajectory driver that materializes every
  $\theta_t$, realized $u_t$, observation identity, reference refresh, and
  proposal metric before estimator replay;
- one shared score/HVP calculation per step, replayed through EMA, AC-only,
  full LFU, and periodic-fresh conditions;
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
5. Run a small paired dense pilot over selected values of:
   - `num_p_steps`;
   - `samples_per_step`;
   - EMA gain or forgetting factor.
6. Include both factorial and approximately compute-matched comparisons.
7. Estimate the compute and storage cost of adding one replica under each candidate configuration.

### Verification gate

- The notebook executes quickly from completed smoke/pilot artifacts.
- Incomplete or incompatible runs fail clearly.
- Every plotted average reports contributing replica count.
- Paired differences are calculated by replica, not by treating trajectory points as independent replicas.
- Assumption-check figures reproduce the stored Phase 3 metrics and make the
  numerical validity and CUDA limitation visible without running model
  computations.
- The pilot reveals whether the chosen step/sample ranges produce distinguishable tracking regimes.

### Check-in decisions

- Select the principal dense `num_p_steps` and `samples_per_step` grid.
- Select EMA gains and reference checkpoint cadence.
- Decide how many initial replicas justify proceeding to EWC coupling.
- Lock the EWC proposal optimizer, inner-step count, and regularization convention for Phase 6.

---

## Phase 6: Dense EWC-coupled experiment

### Goal

Determine whether Fisher-tracking differences change continual-learning behavior when each estimator influences future updates.

### Scope

1. Implement the EWC-regularized proposal $\widetilde u_t$ using the active dense Fisher estimate.
2. Begin with the uncontrolled policy $\pi_t=1$, so

   $$
   u_t=\widetilde u_t.
   $$

3. Clone the shared initialization and paired observation stream into each estimator condition.
4. Allow each condition's Fisher estimate to alter its proposal and future parameter path.
5. Preserve shared non-treatment settings while explicitly recording path divergence.
6. Estimate the high-sample reference Fisher at each condition's own configured checkpoints.
7. Record:
   - data loss and EWC penalty;
   - proposal and accepted update norms;
   - Fisher-weighted drift;
   - old-digit, digit-9, and balanced metrics;
   - Fisher tracking and projection diagnostics;
   - runtime and memory.
8. Add a smoke run covering all dense estimator conditions.

### Tests

- identical Fisher inputs yield identical EWC proposals;
- changing only the Fisher condition can change the path but not the paired data stream;
- each reference Fisher is keyed to the correct condition checkpoint;
- EWC penalties are nonnegative after projection;
- proposal and realized displacement artifacts agree;
- the following LFU uses the realized displacement.

### Verification gate

- All dense conditions complete a small coupled trajectory.
- Divergent paths are evaluated against references at their own parameters.
- Retention and adaptation metrics are reproducible under paired seeds.
- No conclusion depends solely on projected matrices without raw instability diagnostics.

### Check-in decisions

- Decide whether full LFU provides enough tracking or learning signal to justify structured scaling.
- Select the dense conditions and factor settings carried into Phase 7.
- Revise the EWC proposal only if the coupled pilot exposes a documented defect.

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
7. Compare against dense projected checkpoints using:
   - relative Frobenius error;
   - fixed-probe matrix-vector error;
   - EWC quadratic-form error;
   - leading eigenspace error;
   - memory and runtime.

### Tests

- diagonal results equal the diagonal of the corresponding dense operation;
- low-rank-plus-diagonal matrix-vector products match explicit reconstruction;
- rank-zero behavior matches the documented diagonal boundary;
- deterministic seeds reproduce Lanczos outputs within sign/subspace conventions;
- invalid rank and breakdown cases fail clearly;
- the external Lanczos source remains unchanged;
- serialization preserves factors and parameter ordering.

### Verification gate

- Diagonal and each selected rank complete fixed-trajectory smoke runs.
- At least one low-rank setting improves materially over diagonal on a dense-reference metric, or evidence clearly shows that it does not.
- Approximation quality, compute, and storage form an interpretable rank curve.
- Coupled runs are attempted only for ranks that survive fixed-trajectory validation.

### Check-in decisions

- Select ranks for coupled and controller experiments.
- Decide whether the copied Lanczos implementation needs a corrective wrapper behavior or whether a separately versioned replacement is warranted.
- Decide which representation offers the best accuracy/compute frontier.

---

## Phase 8: Optimal-controller experiment

### Goal

Evaluate the adaptation controller after the principal Fisher estimators and representations have been selected.

### Scope

1. Implement controller policies:
   - `uncontrolled`;
   - `fixed`;
   - `optimal_plugin`;
   - `optimal_capped`;
   - `optimal_oracle`.
2. For the plug-in policy, calculate

   $$
   \widehat\pi_t^\star
   =
   \operatorname{clip}_{[0,1]}
   \left(
   1-
   \frac{
   \operatorname{tr}
   [(\widehat{\mathcal I}_t+\lambda I)^{-1}]
   }{
   2n_{\mathrm{eff},t}\|\widetilde u_t\|^2+\varepsilon
   }
   \right).
   $$

3. Apply

   $$
   u_t=\pi_t\widetilde u_t.
   $$

4. Implement damped inverse traces:
   - direct eigenspectrum for dense;
   - elementwise inverse for diagonal;
   - a verified low-rank identity or equivalent stable solve for low-rank-plus-diagonal.
5. Implement a targeted `pi_max` grid for `optimal_capped`.
6. Implement the oracle diagnostic using high-sample reference quantities without presenting it as a deployable condition.
7. Record every term needed to reconstruct the controller decision.
8. Run controller experiments only on the small set of Fisher conditions and representations selected at previous gates.

### Tests

- all policies remain in $[0,1]$;
- zero or tiny proposal norms are numerically safe;
- `optimal_capped` never exceeds `pi_max`;
- `uncontrolled` returns the original proposal;
- accepted updates equal `pi * proposal`;
- LFU consumes the accepted update on the next step;
- dense, diagonal, and low-rank inverse traces agree on matrices representable by all three;
- controller artifacts reproduce decisions exactly.

### Verification gate

- All selected controller policies complete paired smoke and pilot trajectories.
- The oracle condition separates policy quality from plug-in Fisher error.
- Cap activation and its consequences are visible in stored metrics.
- The selected `pi_max` range contains both inactive and meaningfully active regimes.

### Check-in decisions

- Decide whether $\widehat\pi_t^\star$ is useful as a controller, a diagnostic, or neither.
- Select any controller conditions worthy of replication.
- Confirm that controller conclusions are not artifacts of damping or inverse-trace estimation.

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
