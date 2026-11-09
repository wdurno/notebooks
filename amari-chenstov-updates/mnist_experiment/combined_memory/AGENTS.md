# Plan 18 Implementation Contract

Read ../plan18.md before changing scientific behavior. Preserve completed
historical sources and units. All production EWC archives are diagonal vectors.
Gaussian NLL sums 54 coordinates; observations and information clocks count
whole grids. LTM is 3268 coordinates, STM is a 108-coordinate rank-one head.

Choose pi before each four-grid arrival. Freeze solve data, probes, ages,
archive, anchors, and scaling. Transport the old archive along the entire
accepted displacement, then blend in actual evictees only. Recenter the LTM
anchor on eviction only. The observer assimilates incoming scores once;
replay and derivative passes do not advance its clock. No cross-moment C
estimator or discrepancy controller is authorized.

One checkpoint is one atomic accepted batch, including model, archive, FIFO,
controller and metrics. Preserve both q states, source/config hashes, and
cumulative time. Reports exclude incomplete paired blocks and never train.

Use the user's ~/.venv Python. CUDA execution may need the direct execution
tool path because the sandbox hides NVIDIA device nodes. Resource ceiling:
20 cumulative active compute hours, 500 GB including dependencies, 20 GB
actual free reserve. Keep near-top notebook progress and PLAN18_HANDOFF.md
updated. The public report is ../plan18_results.ipynb.
