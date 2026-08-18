# Plan 3 maintenance notes (agents)

This file is implementation context. Human commands stay in
[COMMAND_CENTER.md](COMMAND_CENTER.md), scientific definitions in
[EXPERIMENTAL_CONDITIONS.md](EXPERIMENTAL_CONDITIONS.md), and phase decisions in
[plan3.md](plan3.md).

## Current boundary

- `plan3_command_center` is planning-only. Do not add `prepare` or `run` until
  Plan 3 Phase 0 approves replay semantics and Phase 1 defines a versioned run
  schema.
- `plan3_profiles.json` contains staged intentions, including conditions that
  are contingent on earlier gates. Its `selected` and `memory-matched` replay
  values are cost placeholders, never empirical decisions.
- Reuse Plan 2 replicas 6 through 10 and the exact derived `m=8` bundles. A new
  treatment writes a new immutable run; it never mutates a Plan 2 artifact.

## Replay invariants

- The optimization dataset is current batch plus the buffer as it existed at
  the beginning of the step. The current batch appears exactly once.
- Append current observations only after an accepted optimization step.
- Bounded replay uses deterministic FIFO eviction. Unbounded replay never
  evicts.
- Pure replay drops evictions. Hybrid replay inserts each eviction into the EWC
  archive exactly once.
- The active buffer and EWC archive are disjoint. Track identities in smoke
  tests even if the production archive stores only sufficient statistics.
- Physical MNIST index storage is an implementation convenience. Scientific
  memory accounting charges the logical observation payload and separately
  records actual resident and serialized bytes.

## Cost and pairing invariants

- Count repeated replay presentations as optimizer work, not unique data.
- Separate persistent learner state, peak working memory, process wall time,
  CUDA elapsed time, and offline evaluation cost.
- Common model weights may be separated in incremental comparisons but must
  still be reported.
- Existing Plan 2 no-EWC runs are predictive controls only. They include
  scientific Fisher diagnostics and are not deployment-latency controls.
- Keep stage generation axial. Never expand all replay, LFU, controller,
  representation, and optimizer choices into one Cartesian grid.
