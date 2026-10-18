# Plan 15 Handoff

## Status

**Phase 0 is complete and ready for Sol (2026-10-17).** No production calibration,
predictive pilot, or confirmation cohort has been started. The `development_*` cache roots
are implementation smoke/debug evidence only. Do not pool them with experiments.

The initial implementation lives in `fisher_memory/`. Read its `AGENTS.md` and
[Plan 15](plan15.md). The live, artifact-only report is
[fisher_memory_results.ipynb](fisher_memory_results.ipynb).

## Starting And Resuming

Run from the repository root, using `~/.venv/bin/python`. A persistent terminal
such as tmux is sufficient; the Python scheduler does not depend on an assistant
connection. The user must explicitly allocate production compute before launch.
`<hours>` is a cumulative ceiling across invocations, not additional time.

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.run \
  --config mnist_experiment/fisher_memory/configs/default.json \
  --root cache/mnist_experiment/plan15/production_v1 \
  --allocation-hours <hours>
```

Resume with the same command plus `--resume`. If a STOP file was requested, also
add `--clear-stop`; it is cleared only after identity validation and locking.
Increasing `--allocation-hours` extends the cumulative allowance without changing
scientific configuration. `--max-wall-seconds` limits one invocation;
`--max-units` limits newly completed trajectory units.

```bash
~/.venv/bin/python -m mnist_experiment.fisher_memory.control status \
  --root cache/mnist_experiment/plan15/production_v1
~/.venv/bin/python -m mnist_experiment.fisher_memory.control stop \
  --root cache/mnist_experiment/plan15/production_v1
```

SIGTERM/Ctrl-C are cooperative. An abrupt kill retains the last committed arrival;
at most the current uncommitted arrival is repeated. Pretraining checkpoints at
most every sixteen batches; initial Fisher accumulation checkpoints every chunk.
Never manually delete a completed unit to force reuse. The scheduler resumes
sealed decisions and skips validated completed work.

Default dataset, notebook, and budget locations can be overridden with
`--data-root`, `--notebook`, and `--budget-root`. A run must be a child of its
shared budget root. One writer across that root prevents concurrent studies from
each spending the same 500 GB allowance. Shared datasets, sibling Plan 15 runs,
checkpoint generations, temporary writes, reports, and orphan report exports
are charged. A free-space reserve is enforced separately.

## Scientific Defaults

- Rank-two additive STM; 487 LTM plus 176 STM parameters. No factor reset/merge.
- Replay capacity 32, weight .5; ridge multiplier .1 times mean archived LTM
  Fisher curvature. EWC pi=.025 for rotations, .05 for the mixture.
- Rotations: 120 arrivals of four images, separately linear and sigmoid.
- Mixture: common p=0 arrival, then 100 arrivals of two images through p=.1;
  10.1 expected online 9s. Fixed evaluation prevalence .1 changes evaluation only.
- Initial fit: 30,000 examples, 15 epochs; 1,024 disjoint reference inputs with
  exact model-label integration. Mixture initialization excludes 9s.
- Bounded L-BFGS: 50 iterations/75 closures; two fixed training probes. These
  numerical/strength defaults await Phase 1 calibration, not confirmation tuning.
- Two screening and two fresh validation replicas per environment; nominal
  confirmation target twelve fresh paired replicas and four conditions.

The scheduler performs the bounded common-policy calibration and one permitted
strength-bracket extension, records its choices, and carries a valid diagnostic
candidate even when efficacy is disappointing. It freezes confirmation before
examining confirmation outcomes. No significance-based stopping or extensions.

## Monitoring And Corrections

Status and sealed accepted metrics update during execution. Notebook publication
occurs at startup, at most fifteen minutes apart during progress, and on exit.
Open the existing notebook during a run; the standalone report writer deliberately
cannot compete with either the active Store lock or the shared storage-budget
lock, even for a different run root. It recounts sibling storage before writing.
Its curves distinguish current holdout,
historical holdout, and evicted training cohorts. Detailed diagnostics stay in
small per-unit JSON and CSV artifacts, not notebook output dumps.

Sol owns subsequent execution and repairs. It need not wait for Astra or the user
to fix a defect. Preserve evidence, version materially changed scientific code or
configuration with a new root, document the change and affected results, and do
not silently retain a confirmation label after outcome-informed redesign.
Reporting-only changes do not change the scientific source hash. Training,
curvature, sampler, scheduler, and artifact-code changes do.

The notebook's smoke plots are deliberately poor predictions from tiny initial
fits. They demonstrate software behavior, not a negative scientific result.
Production-shaped timing measurements also use low-fidelity initial parameters;
Phase 1 must replace extrapolations with actual complete-trajectory costs.

## Verification

The immutable acceptance manifest is
`cache/mnist_experiment/plan15/phase0_acceptance_v1/units/phase0-acceptance/acceptance.json`.
It verifies complete artifacts, source/configuration hashes, and pytest XML.

- Focused unit suite: **195 passed**; its four CUDA-only skips subsequently passed
  in a separate **32-test CUDA mathematical suite**.
- CPU integration: **24 passed**, including SIGTERM and SIGKILL during fitting,
  checkpoint writes, validation, and reporting.
- CUDA integration: **16 passed**, with exact interrupted/uninterrupted equality
  for all six conditions on rotations/mixtures and both initialization stages.
- Final CLI smoke: **18 CPU and 18 CUDA trajectories**, all three environments,
  plus a **34-trajectory miniature staged study** through closeout. No nonfinite
  or solver-failure outcomes in these acceptance trajectories.
- Completed-study fresh-process resume: **9.24 seconds**, without retraining.
- Notebook: **15 cells, six figures**, about 0.75 MB; titles/axes visually checked.

Scientific source hash:
`ec987462b89e49239137163c6c52451c134bbc7c94f53576c24377329a6abbfd`.
Exact CPU, CUDA, and pipeline config hashes and the dataset content hash are in
the acceptance manifest. All final run artifacts are under `acceptance_cpu_v2`,
`acceptance_cuda_v1`, and `acceptance_pipeline_v1` inside the Plan 15 budget root.
Reports and control code are excluded from the scientific hash; rendering-only
fixes do not invalidate completed learning trajectories.

## Timing And Remaining Risks

RTX 4070, PyTorch 2.10.0+cu128, CUDA 12.8, float32 execution. The CUDA timing probe
used production dimensions: 32 replay examples, four arrivals, two probes,
50 iterations/75 closures, and the actual 663-parameter model.

| Operation | Measured seconds |
| --- | ---: |
| Replay/EWC solve | 0.438 |
| JVP solve | 1.324 |
| Finite-difference solve | 0.961 |
| Directional-KL solve | 0.947 |
| Distillation solve | 0.657 |
| Four-example eviction Fisher | 0.012 |
| 1,000-image evaluation | 0.0067 |

These are single, low-fidelity-state timing probes, not predictive pilots or
convergence validation. All used the 75-closure ceiling. Phase 1 must inspect
actual initialized-trajectory residuals and under-solving diagnostics; it may
use the planned bounded common-ceiling repair. Initial Fisher diagnostics compare
a nested prefix to the full reference sample, not to independent truth.

Revised **provisional production forecast: about 10--24 active hours**:
Phase 1 0.5--2 h; Phase 2 4--8 h; Phase 3 5--13 h; Phase 4 0.1--0.5 h.
This extrapolates measured solve costs across the planned 12-replica design,
with initialization, audit, serialization/reporting, and repair headroom. Full
production initialization and complete trajectories have not yet been timed.
Twenty hours is a reasonable proposed first allocation, not an authorization
or guaranteed completion time. Replace the forecast after real Phase 1 timing.

The three final smoke roots total about **40 MB**; source datasets are shared.
Production should be in the few-GB range with this artifact design, with a
conservative target below 50 GB. Admission still enforces the hard **500 GB**
ceiling, including all earlier Plan 15 roots and temporary generations.

All Phase 0 compute has stopped. Next action: switch to Sol, agree an explicit
production allocation, and launch a fresh production root with the command above.
