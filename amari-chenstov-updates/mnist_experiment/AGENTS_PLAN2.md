# Plan 2 maintenance notes (agents)

This file records implementation invariants for the low-data boundary search.
Human-facing commands belong in [COMMAND_CENTER.md](COMMAND_CENTER.md), and
scientific condition definitions belong in
[EXPERIMENTAL_CONDITIONS.md](EXPERIMENTAL_CONDITIONS.md).

## Immutable nesting

- `src.plan2` generates exactly three conditions per requested `(replica, m)`.
  Do not reinterpret its `m` list as a general Cartesian search.
- The Phase 9 master is resolved through `phase9_profiles.json`; production
  replicas use `controller-screen:center` at `m=128`.
- A derived bundle may differ from its parent design only in
  `data.samples_per_step`. Its stream is the first `m` ordered observations at
  every `p` step.
- Derived bundles are self-contained copies with `stream_derivation` metadata.
  They retain parent bundle, model, partition, and stream hashes. Never mutate
  the parent or a completed derivative.
- The master and every prefix share initialization and partitions. Conditions
  at one `(replica, m)` share one replica-bundle ID.

## Oracle reuse

- One completed Phase 9 reference-optimum path is reused across all `m` for a
  replica because the model state, parameter layout, partition, and `p` grid do
  not change.
- `run_controller` validates the source run's bundle identity, initial flattened
  parameter hash, parameter count, partition hashes, and path content hash.
  A derived bundle must identify the source run's bundle as its parent.

## Schemas and smoke tests

- Completed coarse-screen configs use config schema 11 and scalar-metric schema
  7. New Plan 2 configs use config schema 12 and scalar-metric schema 8; tensor
  artifacts remain at schema 4. Schema 8 records confusion rates needed for 9
  OvR accuracy, environmental-prevalence-adjusted 9 precision, and 9 recall.
- Never mutate schema-7 runs to add those fields. Derive a content-addressed,
  inference-only classification summary from their saved parameter trajectory
  and fixed holdout partition.
- `plan2_smoke_cpu.json` and `plan2_smoke_cuda.json` cover all three conditions
  at `m=1,2`, using retained replica 9001 and its tiny reference path.
- The canonical command is `python -m mnist_experiment.plan2_command_center`.
  `prepare` derives bundles but never runs initialization or training; `run`
  executes only missing controller trajectories.
