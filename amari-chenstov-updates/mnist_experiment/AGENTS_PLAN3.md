# Plan 3 maintenance notes (agents)

This file is implementation context. Human commands stay in
[COMMAND_CENTER.md](COMMAND_CENTER.md), scientific definitions in
[EXPERIMENTAL_CONDITIONS.md](EXPERIMENTAL_CONDITIONS.md), and phase decisions in
[plan3.md](plan3.md).

## Current boundary

- Phase 7 is complete. Its predeclared fresh-confirmation bundle is
  `plan3-fresh-confirmation__r0011-r0025__89c1451b1983`. Execute replicas in
  complete blocks of five. Replicas 11 through 20 met the precision gate;
  replicas 21 through 25 must remain absent unless the scientific decision is
  explicitly reopened. The accepted analysis is
  `phase7__plan3-fresh-confirmation__r0011-r0025__89c1451b1983__n10__85db75d1349e`.
  The bundle intentionally contains no adaptive or LFU conditions.
- Fresh replicas require two treatment-independent dependencies: an independent
  initialization bundle and a standalone `initial_archive.pt`. The latter is
  a score-only $p=0$ Fisher estimate with the existing adaptive six-sigma
  convergence rule. It must never build or consume a reference-optimum path.
- Phase 7 analysis uses only complete leading blocks and only fresh replicas.
  Underlying runs preserve all 100 linear $p$ steps; summaries and plots alone
  restrict to $p<.5$. Step-$t$ metrics precede its update and therefore have
  exposure `8*t`.
- None of the ten initial Fisher estimates met the 1% six-sigma Frobenius
  early-stop target before the 32,768-score maximum. Do not call these
  converged reference Fishers. The accepted analysis and notebook retain the
  per-replica convergence, dependence, and Lanczos diagnostics.
- Phase 6 is complete. Its bundle is
  `plan3-deployment-frontier__r0006-r0010__f21931201ff8`; the accepted analysis
  is
  `phase6__plan3-deployment-frontier__r0006-r0010__f21931201ff8__ce7440f3fe91`.
  All 35 oracle-free runs completed. Carry fixed Hybrid B32, Replay B32, fixed
  EWC, and unbounded replay into Phase 7. Adaptive policies remain diagnostic.
- Phase 6 hybrid artifacts use configuration schema 16, artifact schema 8, and
  metric schema 12. Analysis schema 2 includes its summary SHA-256 in the
  immutable identity. Never attempt to rewrite the earlier schema-1 analysis.

- Phases 2 through 4 are complete. Capacity 32 is the accepted smallest useful
  bounded replay setting; capacity 128 remains a high-quality bounded
  reference and unbounded replay remains the unconstrained-memory control.
  Phase 4 selected no-LFU hybrid B32 for the Phase 5 LFU isolation. The Phase
  5 bundle is `plan3-lfu-isolation__r0006-r0010__6d8c4ad67265`; it adds EWC
  AC-only, EWC full-LFU, and Hybrid B32 full-LFU trajectories.
- Phase 5 stopped after the three replica-6 production preflights. Directional
  resets occurred on at least 96% of target-region updates; EWC candidates were
  materially indefinite on 92% to 98% of steps, while Hybrid B32 projected
  away 33.5% of candidate Frobenius norm on average. Do not resume the bundle
  without an explicit revised LFU estimand. The gate recommendation is no LFU.
- The accepted Phase 3 bundle is `plan3-phase3__r0006__a8ffdd4d6e7b`. Its
  CPU/CUDA smokes cover capacities 0, 8, 25, and unbounded.
- The accepted Phase 4 bundle is
  `plan3-history-frontier__r0006-r0010__12582f2d297a`; its paired analysis is
  `phase4__plan3-history-frontier__r0006-r0010__12582f2d297a__113026398aa0`.
  Hybrid B8 and B25 remain lower-memory Pareto alternatives. Do not implement
  Phase 5 controls remain immutable and are reused rather than recomputed.
- `plan3_profiles.json` contains staged intentions, including conditions that
  are contingent on earlier gates. Its `selected` replay value is a cost
  placeholder whose value now agrees with the accepted Phase 2 selection.
  Memory-matched replay is derived from the canonical byte contract and must
  never be tuned for predictive quality.
- Reuse Plan 2 replicas 6 through 10 and the exact derived `m=8` bundles. A new
  treatment writes a new immutable run; it never mutates a Plan 2 artifact.

## Replay invariants

- The optimization dataset is current batch plus the buffer as it existed at
  the beginning of the step. The current batch appears exactly once.
- Append current observations only after a successfully completed optimization
  transaction. An exception restores parameters and inserts nothing.
- Source sampling remains with replacement. Repeated MNIST indices are
  distinct arrival events and must not be deduplicated.
- Bounded replay uses deterministic FIFO eviction. Unbounded replay never
  evicts.
- Pure replay drops evictions. Hybrid replay inserts each eviction into the EWC
  archive exactly once.
- A hybrid has a separate archive anchor. The learner may move away from it,
  but only evicted observations update it; still-active observations cannot
  recenter the archive through the learner fit.
- The clean hybrid transaction stages learner fitting, FIFO insertion and
  eviction, archive consolidation, and Fisher replacement, then commits all
  four together. A no-eviction step leaves the archive bit-for-bit unchanged.
- Phase 3 and the Phase 4 baseline use fixed `pi=.05` for learner EWC, archive
  consolidation, and the no-LFU Fisher blend.
- Phase 6 adaptive hybrids checkpoint the controller transactionally. The same
  accepted `pi_t` must weight learner EWC, archive consolidation, and Fisher
  EMA. Deployment artifacts must have no reference path and zero HVPs.
- Schema-v15 hybrid LFUs evaluate gradients and HVPs at the new archive anchor
  using the realized archive displacement. The eight-step directional-ridge
  state is checkpointed transactionally and included in persistent-memory
  accounting. A failed archive update must not mutate the live ridge state.
- The active buffer and EWC archive are disjoint. Track identities in smoke
  tests even if the production archive stores only sufficient statistics.
- Physical MNIST index storage is an implementation convenience. Scientific
  memory accounting charges the logical observation payload and separately
  records actual resident and serialized bytes.
- The fixed-policy EWC summary is `4 * p * (rank + 2)` bytes. A canonical MNIST
  replay item is 800 bytes and FIFO metadata is 24 bytes, yielding capacity 25
  for `p=512`, rank 8. Model and optimizer state are reported separately.
- Fixed `pi=.05` weights the EWC archive against the entire active likelihood
  block. Replay observations are equally weighted inside that block.

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
