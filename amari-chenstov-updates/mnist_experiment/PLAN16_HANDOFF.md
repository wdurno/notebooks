# Plan 16 Continuation

Execution was authorized on 2026-11-01 with 20 cumulative active compute hours.
Plan 17 is not authorized. Read `plan16.md` and `diagonal_lfu/AGENTS.md`.

## Current State

Phase 0 acceptance is complete: focused tests pass, CPU stop/resume reproduces
an uninterrupted learner exactly, and CPU/CUDA smoke studies completed after
timed interruptions. All phases and fresh replication completed on 2026-11-01 in
`cache/mnist_experiment/plan16/production_v3`. The first, incomplete EMA pilot
in `production_v1` was stopped for pre-replication provenance/lifecycle repairs:
consume the recorded model/order seeds, aggregate the action screen over both
development replicas, and handle a stop immediately after a completion seal.
The changes ledger records this revision; no fresh evidence existed in v1.
The partial v2 EMA pilot exposed expensive raw-coordinate solves. Version 3
uses a diagonal rescaling of optimizer coordinates for the identical objective
across every treatment, with physical-coordinate gradient diagnostics. The
saved-solve check changed the objective by only about 2e-7 and left its holdout
accuracies unchanged, while using 11 rather than 75 evaluations. See the v3
changes ledger for equations and evidence. Acceptance roots remain smoke only.

The canonical report is `mnist_experiment/plan16_results.ipynb`. Python entry
points do all training/inference/analysis; the notebook contains compact plots.

## Completion

All 51 production trajectories completed: 15 development and 36 fresh. The fresh
allocation was fixed at eight main and four stationary three-arm replicas before
their outcomes were inspected. Common c=.01 and ridge=.1 remained unchanged.
No fresh pairs are missing. The worker exited and no Plan 16 GPU job remains.

Main-schedule full-path differences versus EMA, in percentage points (nominal
95% paired bootstrap intervals):

| Method | 9 recall | 0--8 retention | Wall-time ratio |
| --- | --- | --- | --- |
| Paired FD | +1.45 [0.05, 2.85] | -3.02 [-3.92, -2.25] | 1.028 |
| Analytic LFU | +0.73 [-0.30, 1.75] | -1.92 [-2.54, -1.27] | 1.074 |

Both show an acquisition/retention trade-off, not an improvement in both.
The stationary cohort has smaller, uncertain recall gains and negative retention
effects; see the notebook's four-row fresh summary. Intervals are exploratory,
unadjusted for multiple comparisons/interim looks, and based on small cohorts.
These equal-exposure comparisons are not retention- or compute-matched and do
not establish sample savings. No efficacy gate or significance stop was applied.

The persisted scheduler clock is 2.425 h including earlier revisions/acceptance;
the family acceptance ledger adds a conservative one-minute post-run charge
for final reports/checks/tests (about 2.44 h accounted in total). Production trajectory
time was 0.50 h development plus 1.58 h fresh. Accounted storage is below 0.7 GB.
All 104 focused tests passed (only PyTorch deprecation warnings); all 71 units
(51 trajectories, 10 assets, 10 initializations) passed full checksum validation.
The production resume command returned `already_completed` without training.

Final canonical report: `plan16_results.ipynb`, with six figures, three compact
tables, explicit undefined-precision annotations, applied conclusions and links
to the sealed snapshot/analysis. Scientific artifacts and earlier revisions are
preserved. No extra action/ridge scan, random-probe or rotation fallback was
needed; Plan 17 is still not authorized. Do not spend the remaining allocation
or start another study automatically.

## Execution And Recovery

Use `~/.venv/bin/python` and an available CUDA GPU. The production/resume command
is:

```bash
~/.venv/bin/python -u -m mnist_experiment.diagonal_lfu.run \
  --config mnist_experiment/diagonal_lfu/configs/default.json \
  --root cache/mnist_experiment/plan16/production_v3 \
  --allocation-hours 20 --resume
```

Do not guess a different configuration or overwrite a source-identity mismatch.
A new scientific source revision requires a new root and explicit lineage.

The runner persists actions, the previous realized parameter endpoint, Fisher
vectors, accepted metrics, RNGs, budget reservations and completion seals.
Use `--resume` after interruptions and `--clear-stop` only when resuming a
previously requested stop. To request a cooperative stop:

```bash
~/.venv/bin/python -m mnist_experiment.diagonal_lfu.control stop --root <study-root>
~/.venv/bin/python -m mnist_experiment.diagonal_lfu.control status --root <study-root>
```

Check `progress.json`, `selection.json`, `runtime.json`, and `report_snapshot.json`
for concise machine state. The foreground scheduler owns progression through
all authorized phases and does not require agent polling or network access.
Never extend the allocation or stop a cohort because of significance.

Completed units are immutable. Preserve all earlier Plan 15 work and other
jobs. Source-writing time is separate; acceptance and experimental computation
are charged to the Plan 16 allocation. No new external-drive migration is needed.

The shared allocation includes acceptance roots and a conservative two-minute
charge for unit tests and derivative checks outside scheduler clocks. Runtime
leases reserve at most one minute against a hard kill. A family-wide lock
prevents two Plan 16 schedulers from spending the same allocation concurrently.

## Resource Pilot

All three corrected v3 pilot trajectories completed on the full 1,312-arrival
schedule at c=.025, ridge=.1. EMA took 117.81 s; Paired FD 130.15 s; Analytic LFU
131.51 s. The EMA solver averaged about 19.3 closures per arrival after scaling.
These are development timings, not matched-compute predictive claims. The
two-replica control screen selected common c=.01 and ridge=.1: larger actions
increased 9 recall but reduced 0--8 retention. All 15 development trajectories
are complete. Before inspecting fresh outcomes, `selection.json` froze eight
main and four stationary replicas, each with all three methods (36 trajectories).
The frozen fresh-cohort forecast was about 71 minutes before its 25% contingency;
the measured fresh trajectory total was 95 minutes, plus initialization/reporting.
Neither an efficacy decision nor a significance stop was made from development.

For an interrupted future extension, if a writer is active, monitor it rather
than launching a duplicate. If it has stopped, use the v3 resume command above
after checking the STOP flag and current
source identity. Earlier v1/v2 roots are retained evidence, not resume targets.

The worker ran in a dedicated persistent tmux server/session after a foreground
stop signal was handled cleanly. That session has exited with the completed
scheduler. Its console output remains in
`cache/mnist_experiment/plan16/production_v3/console.log`. The scheduler and
notebook updates ran without a conversational connection. A user-requested
stop should still use the cooperative stop command, not an automatic restart loop.
