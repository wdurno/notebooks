# Plan 19 Numerical Contract

Version: 1. Owner: Phase 0A/0B numerical slice. Implementation:
`src.continual.picar_math`. No persistence, scheduler, robot, or model loading.

## Model And Coordinates

`Params = Mapping[str, Tensor]`; names and native tensor shapes identify
coordinates, without flattening a backbone. `Predict = Callable[[Params],
Mapping[str, Tensor]]` returns **`return_logits[B,K]`, `action_mean[B,7]`**.
`functional_predict(model, params, args=(), kwargs=None, buffers=None)` uses
`torch.func.functional_call`; supplied parameters override the module's other
fixed parameters. Freeze omitted backbone parameters. Supply deterministic
histories/context, disable dropout, and do not mutate recurrent state or buffers.
The helper snapshots all buffers, including those not explicitly overridden;
it does not change model training mode.
For example, `predict = lambda p: functional_predict(model, p, (histories,))`.

LTM/STM names are explicit, nonempty, disjoint sequences. Each tensor belongs
wholly to one bank. All declared trainable coordinates must be accounted for
by the caller. FD changes only STM coordinates but preserves gradients through
**both endpoints and both banks**. Do not detach LTM inside `predict`.

## Frozen Likelihood

`LikelihoodDesign(bin_values, baseline, temperature, sigma_action,
log_weight_cap=None)` snapshots/detaches inputs. Baseline is `[B]`, bin values
are `[K]` or per-item `[B,K]`, temperature is scalar or `[B]`, and SigmaA is
SPD `[7,7]` or `[B,7,7]`. All are on the baseline device, converted to its
accumulation dtype (float64 stays float64; other floating types use float32).
Use bin midpoints from `return_bin_values(edges)`. Store the design with each
record; never recreate old baselines from current predictions. The optional
log-weight cap is an explicit **upper** cap; `cap_hit_rate` reports its hits.
Underflow/overflow is an error, not an implicit clamp. Tensor fields are
read-only by convention; the constructor protects against input aliasing.

- `joint_nll(output, labels, actions, design) -> Tensor[B]`: complete normalized
  joint categorical/Gaussian NLL, including constants. Labels are int64 `[B]`;
  recorded seven-logit actions are `[B,7]`. Both targets are detached.
- `directed_kl(source, target, design) -> Tensor[B]`: exact source-to-target
  joint KL; source categorical probabilities weight the action term.
- `symmetric_kl(left, right, design) -> Tensor[B]`: sum, not mean, of directions.
- `sample_model_labels(output, design, *, generator) -> (labels, actions)`:
  detached draws from the joint model, **not** exploration Gamma. Caller owns
  the named generator and its saved state; no global RNG is consumed.
- `per_record_scores(predict, params, labels, actions, design) -> dict`:
  detached score tensors `[B,*parameter_shape]`, with score = negative NLL
  gradient. For bounded diagnostics; production Fisher streams the squares.
- `model_fisher_diagonal(predict, params, design, *, method="exact",
  draws=1, generator=None) -> dict`: mean over histories and model labels,
  native parameter shapes. `exact` sums output-Jacobian contributions;
  `sampled` averages squared **individual record** scores across `draws`,
  requiring an explicit generator. Neither allocates a parameter-square matrix.
  Both evaluate a detached parameter snapshot and return detached diagonals.
  Include only desired differentiable parameter tensors (e.g. LTM on eviction).
  Omitted STM coordinates remain fixed at the accepted model state. Exact mode
  is a small-model reference; use `method="sampled"` explicitly in production.

## FD And Objective

- `rademacher_probes(params, stm_names, count, *, generator) -> tuple[dict,...]`:
  iid +/-1 coordinates, covariance identity, no unit-norm normalization.
- `replay_ages(acquired_count, arrival_counts, capacity) -> Tensor[B]`:
  clipped `(acquired_count-arrival_counts)/capacity`; future arrivals fail.
- `stm_fd_penalty(predict, params, design, stm_names, probes, ages, epsilon)`:
  scalar mean over records/probes of `age**2 * symmetric_KL / epsilon**2 / d_S`.
  Endpoints are `theta +/- epsilon * probe / 2`. Probes and ages are detached;
  reuse the same probes and histories throughout an optimizer event. Ages are
  already clipped to `[0,1]`. Empty replay is represented by no FD call.
- `regularized_objective(params, *, ltm_names, stm_names, ltm_anchor,
  fisher_diagonal, stm_anchor, new_nll=None, replay_nll=None, pi=0.025,
  ridge_coefficient=0.0, numerical_floor=0.0, fd_penalty=None,
  fd_strength=0.0) -> dict[str, Tensor]`: keys `total`, `data`, `ewc`, `ridge`,
  `fd`, `fisher_scale`, `kappa`. Data is half each batch mean when both are
  present, otherwise the available mean; providing neither batch is an error.
  EWC is LTM-only.
  Ridge uses `omega*kappa/2` for both LTM-anchor and STM-preupdate distances,
  where `omega=(1-pi)/pi`, `kappa=c_r*max(mean(D), numerical_floor)`.
  `fd` is the strength-weighted penalty. Freeze all anchors across event steps;
  disable implicit weight decay. NLL reporting excludes every regularizer.
- `eviction_ema(diagonal, anchor, accepted, evicted_fisher, *, retired_count,
  pi=0.025, arrival_unit=1.0) -> (new_diagonal, new_anchor, eta)`:
  mappings contain LTM only. Evicted Fisher is the batch mean at accepted
  parameters, using retired items' frozen designs. For positive `e`,
  `eta=1-(1-pi)**(e/arrival_unit)` and anchor becomes accepted LTM. For zero
  retirements, pass `evicted_fisher=None`; both old values remain unchanged.

The caller owns eligibility and exactly-once admission: finish labels and last
permitted replay, accept parameters, evaluate eviction Fisher, then atomically
commit retirement IDs, new diagonal, and new anchor. Repeated replay and sleep
do not create admissions. A kernel invocation is not a retirement ledger.

## Labels And Metrics

- `time_discounts(elapsed, gamma0=0.99, reference_duration=0.5)` computes
  `gamma0**(elapsed/reference_duration)` for positive recorded elapsed times.
- `shaped_rewards(elapsed, potential_start, potential_end, terminal_bonus,
  terminated, gamma0=0.99, reference_duration=0.5)` consumes detached recorded
  inputs and sets terminal endpoint potentials to zero. It implements the
  declared discounted potential difference, elapsed step cost, and bonus.
- `discounted_returns(rewards, discounts, terminated, truncated=None)` accepts
  equal-length 1D arrays of tensors, using discount products and no bootstrap.
  Interior true terminals reset the recursion. The final record must terminate;
  infrastructure truncation anywhere is an error, never a fabricated zero.
- `return_bin_values(edges)` supports `[K+1]` or `[B,K+1]` increasing edges.
- `bin_returns(returns, edges) -> int64[B]`: midpoint bins, left closed and
  right open except the final closed bin. Out-of-range or nonfinite inputs fail.
- `teacher_finite_logits(probabilities, probability_floor)` returns
  `(executed_probabilities, logits)` for `[...,8]` normalized probabilities;
  floor then renormalize, `logits=log(q[...,:7])-log(q[...,7:8])`. Apply **before
  execution**, store original and executed distributions, and use executed
  probabilities for controls and distillation. Hard demonstrations stay tagged.
- `budget_averaged_failure(samples, success)` uses piecewise-linear
  interpolation (trapezoids), from measured sample zero to measured budget N.
- `efficiency_metrics(control_samples, fd_samples, control_seconds, fd_seconds)`
  returns `f`, `c`, `S`. Requires supported matched-capability/retention crossings
  supplied by the caller; no fitted crossing, extrapolation, or censoring policy
  is implemented. Costs must use the same declared operational accounting.

## Numerical Acceptance

Tests are CPU-only, primarily float64, with float32 and reduced-precision output
checks. Accumulation is at least float32; the prediction graph remains attached.
The production wrapper owns autocast and quantization. This slice does not
certify a quantized model's finite-difference resolution or throughput.

Run `OMP_NUM_THREADS=1 ~/.venv/bin/python -m pytest test/unit/test_picar_math.py -q`.
`test_complete_likelihood_fd_ewc_objective_gradient_reference` is the bounded
end-to-end replay-objective reference for adapting a production model wrapper.

Fixtures compare normalized NLL/KL against torch distributions, analytic versus
sampled Fisher, per-record score signs, symmetric FD against exact STM trace and
both-bank finite-difference gradients, zero-retirement anchors, and shaping
telescoping. Efficiency fixtures include 50% samples saved at 2x compute giving
`S=0.25`. Invalid shapes, ranges, covariance, designs, and incomplete returns
must fail explicitly. No LFU, dense parameter Hessian/Fisher, or persistence.

## Revision 2: Fix 2 Dense Visual Reward And Task Sidecar

This additive revision introduces `DENSE_VISUAL_REWARD_VERSION =
"dense-visual-v1"`. The likelihood, score sign, Fisher, FD, joint KL, existing
`CONTRACT_VERSION = 1` and historical `shaped_rewards` behavior are unchanged.
Task/reward identities and the corresponding bin/baseline/temperature/critic
design must be recorded together by the caller for new runs. No historical
labels, completed artifacts, checkpoints or initial Fisher summaries are
modified or silently reused for this new recipe.

### Dense Reward API

`dense_visual_rewards(elapsed, visual_scores, terminal_bonus, terminated)`
accepts equal, nonempty recorded `[T]` tensors on one device. Numeric inputs
must be floating point and finite; `terminated` must be boolean. Output is
detached, on elapsed's device and in its accumulation dtype (float64 remains
float64, otherwise float32). Scores are frozen **post-action** RGB-only
evaluator outputs strictly in `[0,10]`, including on terminal records. They
are validated before dtype conversion; no clipping or fabricated zeros.

The fixed versioned formula is exactly

```
r = 2 * elapsed / 20 * (visual_scores / 10) - .02 * elapsed / .5 + terminal_bonus
```

Elapsed times must be positive; summed recorded duration between true
terminals must not exceed 20 seconds, allowing only floating-point roundoff.
Multiple episodes may share an input batch. Incomplete suffixes may be scored
but not assigned complete returns. With slices, the caller still owns the
full episode clock; the kernel cannot infer an unrecorded prefix or certify
the timestamp of a deadline bonus.

Nonterminal bonuses must be zero. Each terminal must carry exactly one of
`+10` (success), `-5` (unsafe collision), `-1` (deadline). Bonuses are neither
elapsed-weighted nor discounted inside this kernel. The caller owns
exactly-once record admission and the correspondence between terminal bonuses
and authoritative outcomes. Zero-duration repeated terminal captures are not
new reward records. The simulator emits each terminal bonus once, preserving
that state in snapshots both before and after capture.

Use unchanged `time_discounts(elapsed, gamma0=.99, reference_duration=.5)` and
`discounted_returns(...)` to construct complete labels. Do not discount stage
rewards a second time, zero terminal visual scores, or apply potential
differences. `RewardComponents.terminal_potential_zero` is a retained legacy
flag, not an instruction to zero dense visual scores. Infrastructure
truncations remain incomplete, not fake task
terminals. Maximum undiscounted visual exposure is two units per 20-second
episode, independent of picture count; discretized discounted returns need
not be identical after a cadence change.

`dense_visual_return_bounds() -> (-6.0, 12.0)` returns the enclosing range
derived from `-5 - .02*20/.5 = -5.8` and `10 + 2 = 12`. Nonnegative visual
rewards and discounts in `(0,1]` preserve these conservative bounds for all
complete suffix returns. Use existing `return_bin_values` and `bin_returns`;
out-of-range labels still raise. Configuration/orchestration owns the new
41-bin default and bounded 81/161-bin audit, raw differences and ties,
advantage-weight concentration/cap hits, and the log-weight approximation
bound `max_bin_width / (2 * temperature)`. These helpers do not alter the
likelihood or silently replace historical `[-8,11]` designs.

### Simulator Task Sidecar

`SimulatorConfig.task_version` accepts `"safe-stop-v1"` (historical default)
or `"ball-contact-v1"`. The latter uses
`geometry.contact_terminal_event(target_contact, unsafe_contact, elapsed_s,
deadline_s)`: unsafe wall/obstacle contact is collision even when simultaneous
with target contact; otherwise target contact is immediate success; otherwise
the deadline terminates. Contact success precedes deadline on the final tick.
Floor support is excluded; chassis and wheel links both count. There are no
clearance, heading, speed or stopped-hold requirements for contact success.
The drive stops on any task terminal, with unchanged `+10/-5/-1` bonuses.
The original `TerminalGeometry`, `target_geometry`, `in_success_region` and
`terminal_event` retain their schema and safe-stop semantics.

`SimulatorConfig.obstacle_enabled: bool = True` is administrative only. Setting
it false removes both obstacle collision and rendering for the explicitly
separate unobstructed diagnostic. Seeded layout draws, target, arena and drift
are unchanged; this is not an alternate primary-task default. Privileged
`diagnostics()` adds `task_version`, `obstacle_enabled`, `target_contact` and
`unsafe_contact`; the two contact flags report current physical contacts.
They are never added to policy observations, RGB features or timing fields.
Terminal outcomes/bonuses remain the declared hybrid supervision, not a
claim that RGB certifies physical contact.
The existing `SimulatorWorker.diagnostics()` RPC forwards these administrative
fields without adding them to `harvest()` observations or timing fields.

Snapshot schema 1 remains readable: only missing `task_version` and
`obstacle_enabled` default to `safe-stop-v1` and true. `SimulatorConfig.from_dict`
rejects unknown fields, missing historical fields and invalid task/flag values.
Restore verifies the original checksum before normalizing the serialized config
in memory, then requires exact resolved config and run identity, as well as
the existing Bullet/schema compatibility. Task or obstacle mismatches fail.
No saved payload is edited; newly emitted snapshots explicitly include both
fields and retain replay-based solver/contact reconstruction.

Focused CPU acceptance:

```
OMP_NUM_THREADS=1 ~/.venv/bin/python -m pytest -q test/unit/test_picar_fix2_rewards.py test/unit/test_picar_contact.py test/unit/test_picar_math_rewards.py test/unit/test_picar_math.py test/unit/test_picar_math_precision.py test/unit/test_picar_robot_protocol.py
```

Fixtures cover finite/range/temporal/bonus validation, partial intervals,
elapsed exposure budgets, terminal score retention, conservative return bounds,
success-delay/loiter/disappearance/collision arithmetic, retained graded history,
bin-to-weight resolution, contact type/precedence, old geometry, actual CPU
Bullet dynamics and exact replay restore before/after terminal bonus capture.
