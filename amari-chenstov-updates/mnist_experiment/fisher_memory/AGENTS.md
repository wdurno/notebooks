# Plan 15

Read [Plan 15](../plan15.md) and [the handoff](../PLAN15_HANDOFF.md) before running
or changing this experiment. Production execution requires an explicit compute
allocation. Phase 0 smoke evidence is not development or confirmation evidence.

## Experiment-Specific Contract

- The specified model has 487 gauge-fixed LTM coordinates and 176 rank-two LoRA
  STM coordinates. Both blocks learn. The fixed 28x28 input permits deterministic
  7x7 average pooling in place of the equivalent adaptive 2x2 pooling operation.
- There is one likelihood Fisher. STM and LTM matrices are its principal blocks,
  not new estimands. Integrate model labels, not observed-label gradient outer
  products. This experiment's explicit archive condition is an eviction-only
  EMA, without an LFU correction; do not import the legacy LFU scheduler here.
- Replay examples enter EWC only once, on eviction, at the accepted parameters.
  EWC protects LTM only. Ridge has an archive LTM anchor and a pre-arrival STM
  anchor. Never archive a resident example or flush the final buffer implicitly.
- Replay and curvature gradients reach both blocks. Distillation's teacher is
  detached and its auxiliary student is LTM-only; ordinary replay still trains
  both. C's optimizer proposal is detached. Freeze probes within each solve.
- Evaluation and evicted audit inputs never reenter training. Initial, current,
  historical holdout, and evicted training-cohort populations remain distinct.
- Checkpoints atomically bind the learner, exposure ledger, and accepted metrics.
  Completed units are immutable. Scientific source/configuration changes require
  a new run root. Preserve earlier results and disclose materially changed
  mathematics or outcome-informed redesign in the notebook.

## Operations

Use the user's `~/.venv/bin/python`. Keep production work in the Python scheduler,
not notebooks or a conversational loop. Use the shared budget root and its lock;
do not bypass storage admission with direct scientific artifact writes.

Sol may diagnose and repair implementation or mathematical defects without
waiting for a model swap. Record material corrections, isolate affected evidence,
and use fresh confirmation when needed. Poor predictive efficacy alone is not a
stop rule. Never extend a cohort because significance is close.

Refresh the concise notebook through the report module. Keep the full audit trail
in JSON, not in notebook prose. Test changes with the focused unit tests and the
explicit cached-MNIST smoke/recovery tests before resuming affected work.
