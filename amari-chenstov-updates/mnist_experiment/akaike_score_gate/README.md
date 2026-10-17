# Plan 14 Execution

The scientific design is in [plan14.md](../plan14.md). The study uses one
likelihood Fisher, a fixed historical EWC anchor within each evidence episode,
and model-generated labels only for Fisher integration. They are never extra
training examples. All matrix accumulation is float64.

Run or resume the frozen study from the repository root:

```bash
~/.venv/bin/python -m mnist_experiment.akaike_score_gate.orchestrate \
  --config mnist_experiment/akaike_score_gate/configs/default_float64.json \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 --resume
```

Add `--max-wall-seconds 10800` for a bounded invocation, `--max-units N` for a
completed-unit limit, or `--through-phase 2` for a phase boundary. A stop cannot
preempt the current atomic operation. Accepted fits, episode transitions,
source epochs, and reference chunks have resumable checkpoints. A run lock
rejects duplicate writers. Completed units are checksummed and immutable.

Refresh the artifact-only report without starting experiments:

```bash
~/.venv/bin/python -m mnist_experiment.akaike_score_gate.refresh_notebook \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 \
  --notebook mnist_experiment/akaike_score_gate_results.ipynb
```

`configs/smoke.json` and `configs/cuda_smoke.json` explicitly reduce evidence,
replicas, epochs, and references for pipeline testing. Never pool their output
with the default study. CUDA execution must have real device access; there is
no silent CPU fallback. The existing adaptive-pooling CUDA kernel can be
nondeterministic, so interruption validation checks a stated numerical
tolerance rather than promising cross-device bit equality.

The raw 60,000-image training pool is partitioned into 28,000 source images,
2,000 source-validation images, 10,000 online images, 16,000 reference images,
and 4,000 development-evaluation images. Mixture source training excludes
digit 9. Fresh final evaluation uses the independent MNIST test set. This
partition reserves separate source-validation and development panels rather
than borrowing final-evaluation labels.

Policy budgets are fixed profiled-work allowances, not machine kill timers.
Actual online time is reported separately. A denied prediction prevents that
arrival's fitting; a researcher can still evaluate the unchanged model. An
episode that cannot reserve consolidation compute admits no fit and records
the omitted consolidation, retaining the old archive. Neither branch obtains
an uncharged update. High-sample references and shadow fits are research costs,
not deployable policy performance.

When calibrated c=1 makes the uncalibrated comparator identical, it aliases
the same immutable ASG trajectory. The frozen final ledger reports nominal
assignments and this deduplication; it does not create extra replicas.

Phase 0 found a finite-precision optimization plateau in the initial float32
prototype. The main study uses float64 model arithmetic and a tighter
`tolerance_change=1e-12`, while retaining `tolerance_grad=1e-5` and the same
residual certificate for every policy. The original `default` directory and
the `gpu_gate_resume_*`/`precision_audit_*` directories contain preflight
evidence only; never pool them with the main cohort. The paired numerical
check is reproducible with `numerical_audit.py` from its sealed parent states.
