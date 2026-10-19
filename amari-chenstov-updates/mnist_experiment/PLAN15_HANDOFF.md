# Plan 15 Handoff

## Status

**Phases 1--4 completed on 2026-10-18 (America/Los_Angeles).**
`production_v4` contains 284 completed trajectories: 140 development and 144
fresh confirmation trajectories (12 replicas x 3 environments x 4 conditions).
All 630 artifact units passed a full content-hash validation. There are no
incomplete units, recorded solver failures, nonfinite results, or missing
primary retention endpoints. No optional Phase 5 experiment was started.
The run consumed approximately 14.5 cumulative active hours, below its 20-hour
allocation; retired diagnostic roots and implementation tests are separate.
The final production artifacts plus budgeted dependencies occupy about 1.9 GB,
far below the strict 500 GB cap. No experiment process should remain running.

### Final Readout

Positive gains below mean replay/EWC minus selected finite-difference KL NLL AUC.
These are nominal paired-bootstrap intervals for each environment separately,
not familywise claims. Development replicas are not pooled into confirmation.

| Environment | Current gain (95% interval) | Historical gain (95% interval) | Favorable current / historical |
| --- | --- | --- | --- |
| Linear rotation | .13985 [.11385, .17453] | .15995 [.12794, .20097] | 12/12; 12/12 |
| Sigmoid rotation | .13725 [.11127, .17301] | .17728 [.14980, .20575] | 12/12; 12/12 |
| Digit-9 mixture | -.00070 [-.00413, .00336] | -.00010 [-.00578, .00610] | 5/12; 6/12 |

The rotation predictive gain is clear in this bounded-solver experiment, but
STM-to-LTM consolidation is not established. Mean current NLL AUC for the
selected method is .51745/.59207 for linear/sigmoid versus .53250/.61964 for
the frozen initial model; these smaller differences are descriptive, not a
new predeclared confirmatory contrast. Much of the gain versus replay/EWC is
therefore protection against negative adaptation. STM removal often helps the
rotation controls, and reducing that harm is not evidence that useful information
transferred. LTM-only loss improves, but ordinary regularization or reduced
capacity remain live explanations. No fresh-arrival intervention was measured.

Mixture results do not show a clear predictive benefit from curvature reduction.
Distillation lowers LTM-only historical loss but does not establish a predictive
gain in the primary full-model outcomes. Streams contained 4--17 observed 9s
(mean 10.33, expected 10.1); each mixture trajectory's final cohort has shorter
declared follow-up, handled without extrapolation and identically across pairs.

Persistent rotation under-solving remains important: approximately 95--97% of
selected-method updates ended above the gradient tolerance despite the repaired
common ceiling. No failed step was silently accepted. Selected-method mean
online trajectory times were about 203/206/169 seconds versus 114/114/88 for
ordinary replay/EWC. The calibrated larger-ceiling replay control used only
124/126/95 seconds; describe it as an allowance control, not an actual equal-time
comparison. Do not claim computational superiority or isolate consolidation
from these controls alone.

Final focused tests passed 209 cases (four CUDA-only skips), including 59
report tests after the reporting-only readout revision. CUDA recovery evidence from
implementation remains recorded below. The concise notebook includes a frozen
reference, a plain-language readout, measured cost, and these interpretation
caveats. A reporting-only refresh can replay this completed root with the resume
command below; it must not create or replace scientific trajectories. The
scientific source hash remains `15b2fd8224d22ebca6391f7e0c2ba81031fa98205df62484a97624e4d8f58c91`.

### Execution History

**Production phases 1--4 started under Sol on 2026-10-17.** The user authorized
the full staged execution after Phase 0. The initial cumulative active-time
allocation is 20 hours, following the proposed allocation below. The run root is
`cache/mnist_experiment/plan15/production_v4`; its status, sealed decisions, and
checkpoints are the authoritative progress record. The scientific source hash
was versioned after the common-solver correction below. GPU access requires execution outside the restricted
sandbox on this node. Resume this root with the following command:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.run \
  --config mnist_experiment/fisher_memory/configs/default.json \
  --root cache/mnist_experiment/plan15/production_v4 \
  --allocation-hours 20 --resume
```

Add `--clear-stop` only when resuming a requested cooperative STOP. Recheck that
no writer owns the root before launching; do not launch a second process after
an assistant disconnection while the first is still running. No change to the
confirmation cohort has been made at launch.

The first attempt, `production_v1`, stopped cleanly after about two minutes.
Its complete default linear trajectory had 100% under-solved updates (gradient
infinity norm above 1e-3), but the scheduler had only triggered the planned
100-iteration/150-closure repair for outright solver failures. Sol corrected
that omission: a majority of under-solved updates now also triggers the one
bounded common-ceiling repair. All later treatments inherit the selected common
ceiling. This changes scheduling, so `production_v1` remains separate diagnostic
evidence; do not resume or pool it with later roots. Significant repair notes persist
across resume and appear in the live notebook. No confirmation data was seen.

`production_v2` also stopped cleanly during Phase 1. The larger ceiling reduced
the default linear mean final gradient from .01375 to .00108, but several rejected
line-search increases near convergence were classified as solver failures.
Version 3 retains the previous accepted iterate for every objective increase,
records the rejected increase and its tolerance, and calls increases no larger
than 16 machine epsilons times max(1, absolute objective) rounding stagnation.
Material increases remain solver failures. CPU float32/float64 regression tests
check both cases. Policy search cannot reselect the original shorter ceiling
after applying its repair. Old development artifacts remain separate; no confirmation
outcomes informed either repair.

Version 3 completed Phase 1, but its mixture directional-KL check falsely rejected
one checkpoint because the fixed local radius .001 had 4.3% gradient-magnitude
truncation error. The independent derivative check passed. On the same saved
checkpoint and direction, radius .00025 reduced that error to 1.08%, with cosine
above .99997. Sol added the bounded mathematical-check ladder
{.001, .0005, .00025, .0001}, recording every attempted radius. All checks must
still pass; failure at every radius leaves the method unavailable. The actual
training direction and radius cap are unchanged. Version 3 stopped cleanly in
early Phase 2, before any confirmation, and remains separate. Version 4 reruns
calibration so the mixture directional method can receive a fair predictive
pilot if its mathematical checks pass. The repeated calibration costs about
half an hour, documented here rather than silently pooling versions.

Version 4 scientific source hash:
`15b2fd8224d22ebca6391f7e0c2ba81031fa98205df62484a97624e4d8f58c91`.
The updated focused suite passed 206 tests (four CUDA-only skips in the CPU
invocation), and the miniature staged scheduler completed through closeout.
The learner corrections also passed all 16 CUDA recovery cases. Verification
records are `repair_v4_unit_tests.xml`, `repair_v3_cuda_recovery.xml`, and
`repair_pipeline_v4` under the Plan 15 cache. The first seven repeated baseline
trajectories matched version 3 exactly on current, historical, prequential, and
initial NLL AUC and realized positive counts. Source/configuration identities in
the production root, rather than this prose, determine resume compatibility.

**Version 4 Phase 1 is complete; Phase 2 is running.** Calibration took about
25 active minutes. All A/B/C methods and distillation passed in all three
environments. The common policy is replay weight .5, ridge multiplier .1,
100 iterations / 150 closures. Two probes are used; the checked FD step is .01
for both rotations and .005 for the mixture. The mixture baseline passed the
acquisition/retention health checks; rotation baseline balance remains unresolved
and is disclosed. The last mixture directional check passed at local audit
radius .00025. No confirmation results exist yet. Later status/ledger artifacts
supersede this checkpoint prose.

Early Phase 2 interpretation question, not a conclusion: the two-replica linear
JVP screen improved mean current/historical NLL as strength increased. At the
largest tested coefficient (53.63), means were .573/.544 versus .800/.810 for
replay/EWC. However, pre-eviction STM reliance was already nearly zero, and
muting STM often helped the unregularized control. Capacity suppression or
ordinary sensitivity regularization is therefore a live alternative to memory
transfer. Revisit the paired pre/post-eviction functional diagnostics at closeout;
do not infer consolidation solely from lower post-eviction reliance. Validation
and confirmation remain pending; no extra control is automatically authorized.

The same screen's frozen-initial reference is even better: mean current NLL AUC
.529 versus .565 for the strongest directional setting. Rotation improvements
over adapting replay/EWC may therefore be protection against negative adaptation,
not useful acquisition. Report both comparisons at closeout and keep the mixture
test central. A reporting-only update adds the stored frozen-model NLL as a
dashed curve and this capacity-suppression caution to the notebook. It passed
57 report tests; the scientific hash stayed unchanged. A clean checkpointed
restart at about 2.26 cumulative active hours loaded the renderer in v4.

At about 3.08 active hours, linear Phase 2 completed (54 total trajectories,
zero solver-failure units). The sealed linear winner is finite-difference KL,
coefficient 53.633903847187014, h=.01, two probes. Fresh validation means were
current/historical .561766/.493613 versus .660280/.634098 for replay/EWC;
the frozen-current reference was .615151. This is two-replica development,
not confirmation. JVP and directional KL also had favorable mean validation
gains; distillation was slightly worse than replay/EWC. Linear compute-matched
replay is frozen at 195 iterations / 292 closures, using the maximum paired
development fit-time ratio 1.94099. Sigmoid Phase 2 is now running independently.

Interpretation clarification: `pre_eviction` is after rehearsal, immediately
before the cohort's first member is evicted. It is not fresh-arrival STM use.
Low reliance there is compatible with the proposed transfer mechanism; it does
not establish that STM was unused from the start. Capacity suppression remains
an alternative, not an established explanation. The reader's glossary and
functional caveat in the report source now say this explicitly. That wording
will load on the next resume, or should be included by a fresh artifact-only
report refresh at closeout; the current process imported the preceding renderer.

Resumed on 2026-10-18 after node preemption at 5.06 cumulative active hours.
The writer lock was free; CUDA was available. The interrupted sigmoid
distillation trajectory recovered its accepted step 98 and continued without
changing scientific identities. This resume also loaded the glossary wording
above. Sigmoid Phase 2 subsequently sealed finite-difference KL, coefficient
26.88965972439316, h=.01, two probes. Its fresh two-replica validation means were
current/historical .598606/.522949 versus .705795/.687189 for replay/EWC;
the frozen-current reference was .718449. Compute-matched replay is fixed at
193 iterations / 290 closures (development fit-time ratio 1.92840).
At about 6.67 active hours, the mixture screen is running, 112 trajectories are
complete, and no completed unit has a recorded solver failure or nonfinite
result. These are development results, not confirmation. The 20-hour cumulative
allocation, fixed 12-replica confirmation target, and scientific hash are
unchanged. Live status and sealed decisions supersede this prose.

**Phase 2 completed; Phase 3 started at about 7.97 active hours on 2026-10-18.**
All three environments selected finite-difference KL. The mixture coefficient is
.0017778404607752718, h=.005, two probes. Its fresh validation current/historical
means were .972700/.791828 versus .965603/.784297 for replay/EWC: the sealed
classification is `diagnostic_no_acquisition_gain`, not a favorable candidate.
It still receives the planned confirmation rather than stopping for efficacy.
Mixture distillation uses .005090545506809547; compute-matched replay uses
199 iterations / 298 closures (development fit-time ratio 1.98548).

The sealed confirmation ledger fixes 12 fresh replicas per environment and
144 trajectories total: ordinary replay/EWC, selected FD, distillation, and
compute-matched replay. Jobs are round-robin by replica/environment. Its
outcome-independent forecast is 29,310.6 seconds (8.14 hours) remaining,
including 20% timing overhead, and 3.27 GB projected artifacts with byte
headroom. The 20-hour active-time allocation is sufficient on that forecast.
Do not alter the cohort from running outcomes or release its inference early.

**At the Phase 0 handoff (2026-10-17),** no production calibration,
predictive pilot, or confirmation cohort had been started. The `development_*` cache roots
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
  --root cache/mnist_experiment/plan15/production_v4 \
  --allocation-hours <hours>
```

Resume with the same command plus `--resume`. If a STOP file was requested, also
add `--clear-stop`; it is cleared only after identity validation and locking.
Increasing `--allocation-hours` extends the cumulative allowance without changing
scientific configuration. `--max-wall-seconds` limits one invocation;
`--max-units` limits newly completed trajectory units.

```bash
~/.venv/bin/python -m mnist_experiment.fisher_memory.control status \
  --root cache/mnist_experiment/plan15/production_v4
~/.venv/bin/python -m mnist_experiment.fisher_memory.control stop \
  --root cache/mnist_experiment/plan15/production_v4
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

All Phase 0 compute has stopped. The original handoff checkpoint was followed by
the user-authorized production launch recorded at the top of this file.
