# Plan 18 Phase 9E

Production now continues through the versioned scheduler correction. Resume
with:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.adaptive_f1_scheduler.run --resume
```

Its root is `cache/mnist_experiment/plan18/adaptive_f1_v2`. It references the
completed v1 units and resumes their accepted state; do not clear v1's STOP
or restart its old scheduler. The numerical source component below is frozen.

## Original Numerical Entry Point

From the checkout, use the project's virtual environment:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.adaptive_f1.run --resume
```

The immutable root is `cache/mnist_experiment/plan18/adaptive_f1_v1`.
`--stage status`, `--stage stop`, and `--stage report` inspect, cooperatively
stop, or render existing scalar artifacts. After a STOP, resume with
`--resume --clear-stop`. Full execution requires matching numerical/recovery
acceptance and a passing independent R/K development retention gate. The
original six-condition calibration remains closed.

The shared Phase 9 cap is 20 active hours, including the earlier 4.36 hours,
all smoke/acceptance work and this extension. The controller uses the existing
diagonal observer approximation with non-diagonal EWC; no LFU is implemented.
Report: `mnist_experiment/plan18_results.ipynb`. CPU/CUDA smokes require separate
`--root` and `--notebook`, plus `--smoke --replicas 2`.
