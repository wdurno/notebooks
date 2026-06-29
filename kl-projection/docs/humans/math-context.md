# Mathematical Context

This experiment inherits mathematical motivation from the Amari-Chentsov update work copied into `/notebooks/imported/`.

The useful positive result is the single-observation batch perspective:
online sufficient-statistic updates can reduce activation-memory pressure while preserving a coherent continual-learning interpretation.
This is relevant to phase 3, where QLoRA, replay, EWC, and Actor-Critic learning may be combined.

The useful negative result is just as important:
direct Amari-Chentsov tensor updates require numerical operations that are too expensive in the naive implementation.
For this project, that means phase 1 and phase 2 should not depend on Amari-Chentsov updates.
Phase 3 may use lower-rank Fisher/EWC mechanics and forgetting diagnostics, but only after the phase 1/2 hierarchy works.

