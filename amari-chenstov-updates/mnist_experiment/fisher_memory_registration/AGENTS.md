# Plan 15 Phase 10

Follow the root/local instructions and Phase 10 in ../plan15.md. The user
authorized implementation and execution on 2026-10-26 under the single 20-hour
active-compute allocation and shared Plan 15 500 GB peak artifact cap.

- Inputs are two 28x28 views of one MNIST source; predict physical-pixel
  displacements of three virtual anchors (six outputs), or its first coordinate.
- Use the same nontrivial affine distribution in nominal pretraining, streams
  and holdouts. Source IDs, affines and branch/strength schedules are paired.
- The continuous sampling law starts at the initial distribution; original
  group indices survive novel-event splitting. Labels stay unchanged by blur.
- Gaussian covariance is 4 I. Sum coordinate NLLs per example; integrate labels
  under the model for Fisher J.T J / 4, not observed residual outer products.
- Both adaptive methods share capacity256/replay16 and all trainable parameters.
  Retirement happens once per exposure. Never mutate a completed unit.
- Rendering and scalar/vector means use distinct schemas from old angle models.
  Do not edit frozen scientific packages or bypass their source checks.
- Sol owns corrections, validation, autonomous scheduling, reporting and
  closeout; record material changes with new immutable identities and lineage.
- The notebook is artifact-only, concise, chronological and refreshed during
  execution. Preserve all historical cells. Resume skips sealed work and retains
  cumulative time/storage usage; no agent connection is required for scheduling.
