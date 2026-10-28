# Plan 15 Phase 9

Follow the root/local AGENTS and Phase 9 in `../plan15.md`. This is a new
continuous-rendering protocol, not a revision of frozen Phase 8/8.1 results.

- Preserve the 422-coordinate Gaussian angle-regression likelihood, fixed
  45-degree scale, model-target Fisher, FD KL, replay/EWC and retirement clock.
  Rendering changes the inputs, not the likelihood or Fisher estimand.
- Every source role uses Uniform[-60,60] applied-angle targets. Rotate first,
  then gamma, then normalized separable Gaussian blur with fixed radius six,
  zero padding and continuous sigma. Zero strength is exactly nominal.
- Contexts share initialization, IDs, angles, digit/domain draws, stream order,
  split segments, evaluation strengths and named numerical seeds. Strip the
  first `--` suffix from stage families before deriving any paired seed.
- Keep the exact drip law and novel-event checkpoints. Rendering strength uses
  the original one-based nominal group even after splitting that group. Domain
  membership is audit metadata, never a learner input.
- Initialization, validation, reference Fisher, stream and diagnostic sources
  are disjoint. Fixed development/confirmation test pools never supply training
  gradients, Fisher inputs or probes. Diagnostic fits are separate extra-label
  learnability checks, never online initial states or competing online arms.
- Cache nominal CPU tensors once. Contextualization shares pretraining,
  validation, reference and nominal panels, changing only novel renderings.
  Persist compact IDs/angles/strengths rather than repeated rendered datasets.
- Plan all epoch seeds through `max_init_epochs` before the common stage is
  chosen. Preserve each 15/30/45-epoch initialization version and its lineage.
- Keep capacity 256, replay counts 4/16, the prospectively frozen one-degree
  retention ceiling and 0.2-degree tolerance. Refinement is at most eight new
  settings per context; resumption never replenishes it.
- User authorization covers Phase 9 execution. The scheduler owns the single
  persistent 20-hour active-compute ledger and shared 500 GB storage cap;
  data/config helpers do not reset or duplicate those budgets. Completed units
  and scientific decisions are immutable, and confirmation cannot reopen them.
