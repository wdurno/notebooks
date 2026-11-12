# Plan 18 Phase 9E Continuation

From the checkout:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.adaptive_f1_scheduler.run --resume
```

Default destination: `cache/mnist_experiment/plan18/adaptive_f1_v2`.
Default read-only parent: `cache/mnist_experiment/plan18/adaptive_f1_v1`.
The parent must be stopped. Its completed artifacts remain in place, and its
accepted incomplete checkpoint is adopted under the new identity. Both writer
locks are held while running. Numerical sources and settings are unchanged;
the separately hashed scheduler uses the measured first-pair bracket for the
one allowed final repair and avoids revalidating complete recipes.

Use `--stage status`, `--stage stop`, or `--stage report` on the same command.
After a stop request, add `--clear-stop` to `--resume`. Do not clear the parent
v1 STOP. The original Phase 9 20-hour allowance covers both roots and all
acceptance/test work; this continuation adds no compute allowance.
