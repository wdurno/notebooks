# Agent-only Phase 9 maintenance notes

This file is implementation context for future agents, not user documentation.

- `phase9_profiles.json` is the source-controlled experimental intention.
- `src.phase9` expands a named profile/replica selection into an immutable
  bundle under `cache/mnist_experiment/phase9/bundles`.
- One `optimal_plugin` entry per replica design owns the reference path. Every
  dependent run names that anchor artifact in its frozen schema-v10 config.
- Controller-axis treatments share one fixed `pi=0.05` control per replica.
  Data-axis cells receive separate controls because their streams differ.
- Overlapping bundle manifests are legal. Analysis deduplicates exact run IDs
  and rejects conflicting metadata so evidence is never counted twice.
- The runner skips completed initialization and controller artifacts, requires
  explicit resume for incomplete runs, and never mutates completed runs.
- Add new search regions as named axial profiles or cells. Do not make a hidden
  Cartesian product or add an independent Fisher forgetting hyperparameter.
- Keep `results.ipynb` artifact-only. Detailed diagnostics belong in the three
  supplemental notebooks.
