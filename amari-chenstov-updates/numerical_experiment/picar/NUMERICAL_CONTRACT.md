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
