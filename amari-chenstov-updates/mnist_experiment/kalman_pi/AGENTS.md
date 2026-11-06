# Plan 17 Execution

Read `../plan17.md`. Execution is authorized on 2026-11-01, with 20 cumulative
active compute hours. Preserve Plan 16 and all historical scientific sources.
This study compares action policies, with direct-EMA diagonal Fisher updates
for every learner. Do not introduce LFUs, replay training, STM, or FD KL.

Choose each action before reading its batch. Observed-label scores at a common
anchor supply the observer signal; model-label-integrated scores supply Fisher
curvature. They must never be substituted for one another. Filter calculations
and score moments are float64. Record uncertainty and mean-motion separately.
The clock advances on stationary holds, not just on environmental movement.

Reanchoring translates the discrepancy and preserves velocity/covariance. It
refreshes the anchor metric/noise normalization with already available training
inputs, without assimilating repeated evidence or advancing nominal weights.
Never use evaluation data for calibration, anchor refresh, or policy choices.

Use immutable units, explicit source identities, accepted-arrival checkpoints,
and a family allocation lock. Preserve all RNGs, filter/noise/anchor state and
nominal weights on resume. Material repairs require a new computational root.
Keep notebook progress near the top and the final accuracy/cost table concise.
Publish all outcomes and trade-offs; no significance stopping or efficacy gate.

Phase 5 execution is authorized on 2026-11-02. Its independently calibrated
ridge study lives in `retention/`, keeping the historical scientific manifest
unchanged. Development holdout accuracies may select ridge as specified in
the plan; confirmation panels may never select settings. The prohibition on
evaluation calibration above applies to observer noise/live actions. Retain
the unchanged pi=.025, ridge=.1 reference, +/-0.5 pp development matching band,
frozen observer settings, shared 20-hour ledger and 500 GB storage ceiling.

Phase 5E was separately approved on 2026-11-02: eight new changing replicas,
fixed/.1 versus fixed/0 versus Kalman/0, within the same family cap. Its nested
`retention/supplement/` code must not alter primary Phase 5 selection or sources.
Preserve primary data-role partition seeds while giving all new replica draws
their own namespace. Run after the primary finishes, seal the three-arm design
before learning, and report all three paired contrasts and both accuracies.
