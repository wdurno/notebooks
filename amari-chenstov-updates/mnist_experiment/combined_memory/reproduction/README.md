# Plan 18 Phase 8

Run from the checkout with the existing virtual environment:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.reproduction.run --resume
```

The default production directory is
`cache/mnist_experiment/plan18/stm_reproduction_v1`. The source-matched
acceptance record is required before production. The command performs the
historical checks before any fresh trajectories. Completed units are reused;
there is no calibration or automatic expansion of the fixed cohort.

```bash
~/.venv/bin/python -m mnist_experiment.combined_memory.reproduction.run --stage status
~/.venv/bin/python -m mnist_experiment.combined_memory.reproduction.run --stage stop
~/.venv/bin/python -u -m mnist_experiment.combined_memory.reproduction.run --resume --clear-stop
~/.venv/bin/python -m mnist_experiment.combined_memory.reproduction.run --stage report
```

`--stage fidelity` runs only historical acceptance. The notebook is
`mnist_experiment/plan18_results.ipynb`. `report/latest.json` points to an
immutable report generation. Reports load scalar artifacts and never train.

Tiny pipeline checks use a separate root and notebook:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.reproduction.run \
  --smoke --device cpu --replicas 2 \
  --root cache/mnist_experiment/plan18/phase8_smoke_cpu \
  --notebook cache/mnist_experiment/plan18/phase8_smoke_cpu/results.ipynb --resume
```

Production uses eight paired replicas, 24 blurred and 16 no-blur paths.
The three same-seed historical paths are separate acceptance evidence.
The original 100-event clock is only a matched schedule in the no-blur control;
that control reports predictive risks and never novel-sample savings or S.
