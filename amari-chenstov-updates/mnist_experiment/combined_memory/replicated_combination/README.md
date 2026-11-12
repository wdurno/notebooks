# Plan 18 Phase 9

Use the project's virtual environment from the checkout:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.replicated_combination.run --resume
```

The production root is `cache/mnist_experiment/plan18/replicated_combination_v1`.
Status, stop and report commands use `--stage status`, `--stage stop` and
`--stage report`. A cooperative STOP is cleared only by `--resume --clear-stop`.
Notebook: `mnist_experiment/plan18_results.ipynb`.

Production requires matching acceptance, full F1 fidelity and a passing
four-member retention-calibration seal before launching eight fresh six-policy
blocks. R alone uses rank-64-plus-diagonal EWC. B/R keep ridge one; LFU is
diagonal-only. New sources/configurations require new immutable run roots.

CPU/CUDA smoke runs use `--smoke --device cpu` (or cuda), a separate `--root`,
and a separate `--notebook`. Smoke cohorts cannot contribute efficacy evidence.
