# Plan 15 Handoff

## Phase 7E Efficiency Conclusion (completed)

User approval and execution were on 2026-10-24. This was a CPU-only,
artifact-based follow-up, not more training or confirmation. The final focused
unit/recovery/report suite passes 111 tests. New implementation lives outside
the frozen scientific package in `plan15_efficiency.py` and
`plan15_efficiency_report.py`. The original scientific source hash below is
unchanged. The existing report received only a presentation-order change to
keep this conclusion after refreshed Phase 7 plots.

Analysis root: `cache/mnist_experiment/plan15/regularization_efficiency_v1`.
Its sealed plan retains the fixed configuration, named bootstrap seed,
2,000 whole-paired-replica index sets, original 7D inference, and completion
seals for all 201 development/confirmation units. Separate sealed units retain
the analysis/draws and visual fit review. Completed-work resume skips fitting
and bootstrap; unfinished bootstrap work can safely repeat. The notebook adds
exactly three cells at the end of Phase 7; all 38 earlier cells are unchanged.

**Conclusion:** the original budget-averaged gain remains **0.13017 nats**, with
its original paired 95% interval **[0.10456, 0.15706]** and 12/12 favorable fresh
replicas. Neither the primary interval nor p-value was recomputed. The new
independent-uniform-budget interpretation does not need monotonic learning.
Retention cost, 1.98-times online compute, frozen-reference gain and unbracketed
ridge-grid boundaries remain visible.

**Exploratory sample equivalence:** separate OLS fits on mean rotated-test NLL
use all events 1--100, excluding initialization. Replay's fitted endpoint
target is 1.3190592 nats. FD's model crossing is **43.2493** images, implying
**56.7507 images (56.7507%)** less than replay's 100 under that approximation.
The diagnostic captures the overall decline but has early overshoot and local
reversals; the sealed visual review explicitly qualifies the fit. This is not
mean first-passage time, guaranteed future saving, or STM-to-LTM identification.

**No ordinary 95% inversion interval:** bootstrap draws 848 and 1451 already
meet the fitted replay target at initialization; 2/2,000 are inversion-invalid
under the predeclared guard. All draws and raw crossings are retained. There
are zero out-of-range finite crossings. Do not quietly discard these two draws
or relax the guard to publish an interval. Any changed target/model is a new,
explicitly post hoc analysis, not a correction of the sealed analysis.

Tested completed-work resume, from the repository directory:

```bash
~/.venv/bin/python -u -m mnist_experiment.plan15_efficiency \
  --source-root cache/mnist_experiment/plan15/regularization_confirmation_v1 \
  --root cache/mnist_experiment/plan15/regularization_efficiency_v1 \
  --resume
```

An interrupted job uses the same command; add `--clear-stop` only when explicitly
resuming after a deliberate STOP. The shared budget lock, 500 GB study cap,
100 MB extra-analysis allowance and original 26-hour Phase 7 allocation remain
enforced. Existing fitting/draw/result units and scientific dependencies are
never rewritten. Notebook rendering only reads sealed results; it does not
train, download, evaluate models, or bootstrap. Phase 6 remains deferred.

Final checks: completed-work resume ran with fitting/bootstrap explicitly
disabled and passed; all analysis seals, all 201 original scientific seals,
and all 38 earlier notebook cells stayed unchanged. The diagnostic legend was
visually checked after a rendering-only clipping fix. Final focused suite:
111 passed, plus five targeted renderer checks. The four recorded test
invocations total 130.17 CPU process-seconds, charged once in a sealed
acceptance unit. Including analysis/reporting/recovery validation, 7E adds
approximately **2.4 CPU minutes**, **4.2 MB** of artifacts, and no GPU work;
aggregate Phase 7 expenditure is approximately **9.48 hours**. A separate
append-only clarification corrects a hash-label typo in the acceptance record;
the plan's dependency/source identities and all scientific results are unchanged.
All Phase 7 work is complete; no experiment process remains running.

## Phase 7 Execution (completed)

The new Phase 7 section of [Plan 15](plan15.md) specifies four sub-phases for
Sol to implement and execute: 7A frozen-objective solve audit,
7B paired ridge grid, 7C bounded FD retuning, and 7D fresh confirmation/reporting.
Phase 6 remains provisional and is neither a dependency nor an automatic next job.
The user authorized 7A--7D on 2026-10-23 with the planned initial 26-hour
aggregate allocation. Implementation and CPU/CUDA recovery acceptance passed:
389 tests with no skips/failures, plus a 41-test reporting follow-up. The sealed
manifest is `cache/mnist_experiment/plan15/phase7_acceptance_v1/units/phase7-acceptance-v1/result.json`.
Its 397.837 measured test-process seconds are charged to the shared allocation.
All four phases completed on CUDA on 2026-10-24, with the final cohort sealed
at 12:42 UTC. No Phase 7 work remains pending and no experiment process remains
running. Phase 6 has not started. Scientific source hash:
`26ca52703bfcc4d13f16d1d42b53851f2d31e443aef8bf035f06056f51a64e75`.
Do not change scientific source files in place while this root is active;
use the recorded correction/versioning protocol if a defect needs repair.
The final notebook includes a brief ridge/FD glossary paragraph and labels
the after-7A forecast as a historical estimate, not a live countdown. It also
flags selected ridge-grid boundaries as best tested, without a bracketed optimum.
The collapsed diagnostics also show recorded fitting, eviction, evaluation,
and total online seconds, plus a concise interpretation of the final intervals.
The final 43 focused reporting tests passed, and the scientific source hash
remains unchanged. These reporting-only clarifications have been published
through the tested completed-work resume path, with the final plots inspected.

**Final confirmation:** all 48 trajectories / 12 paired replicas are sealed.
At $c_R=c_F=1$ and $\lambda_F=160.89$, FD's primary acquisition gain over replay
was +0.13017 (paired bootstrap 95% interval [+0.10456, +0.15706], one-sided
sign-flip p=0.00024414), with 12/12 favorable replicas. Mean acquisition NLL
was 1.42354 for replay, 1.29336 for FD, and 1.41341 for the frozen reference.
FD's nominal acquisition gain over freezing was +0.12004 [+0.05893, +0.18800],
with 11/12 favorable replicas; replay's was -0.01013 [-0.07069, +0.05102].
Mixed-stream base retention harm, FD minus replay, was -0.03034
[-0.03716, -0.02438], but FD still worsened base acquisition NLL from
initialization by +0.06461 [+0.05881, +0.07094]. This is an adaptation/retention
trade-off, not forgetting-free learning. Endpoint rotated accuracy was 69.36%
for FD versus 66.46% for replay, descriptively.

Mixed online times averaged 220.85 seconds FD versus 111.42 seconds replay
(1.98 times). Equal-data evidence is not equal-compute superiority, a direct
samples-saved estimate, or a transfer-mechanism identification. Solver failures
and nonfinite trajectories were zero; 7,583 residual-based under-solved updates
remain disclosed. The full fixed-cohort analysis reproduced exactly on resume.
All 201 development/confirmation units passed full checksum validation, and
every completion seal stayed unchanged through standalone and all-stage resume.
Both store writer locks were released. Total active compute was approximately
9.44 hours including acceptance and reporting, with about 0.78 GB stored across
the two scientific roots and acceptance root. Stronger ridge made solves faster
than the inherited timing forecast; the unused allocation is not new scope.

**Development history:** all sixteen initial trajectories and 48 frozen
objective audits sealed successfully. Two audits crossed the predeclared
objective-gap threshold, below the twelve needed for a repair. The sealed
common policy remains 100 iterations / 150 closures; no repair cohort ran.
The 7B ledger covers 64 trajectories, reusing the sixteen compatible 7A units.
Measured online means were 238.05 seconds for replay/EWC and 428.87 seconds for
FD. The sealed timing forecast is 12.42--19.44 active hours remaining after 7A,
within the cumulative allowance. Development storage was about 395 MB at that
transition. The live status and later sealed decisions supersede this snapshot.

The full 64-trajectory ridge grid sealed on 2026-10-24. The immutable shortlist
selects $c_R=c_{F,0}=1$, with both candidates retention-feasible under the
predeclared common reference. Mixed-stream development acquisition NLL means
were 1.30839 for replay and 1.19176 for FD at the inherited coefficient 160.89;
the mean frozen reference was 1.24016. These four-replica development outcomes
are descriptive, not confirmation. The strongest tested ridge won for both
methods, so the optimum is not bracketed. Do not expand the ridge grid.
Phase 7C compared FD coefficients 53.63, 160.89 (reused), and 482.67 at
that one ridge. Their mixed acquisition means were 1.20957, 1.19176, and
1.20631, respectively. The middle coefficient won, so the sealed extension
decision requests no extra coefficient. Final recipes are replay at $c=1$
and FD at $(c,\lambda_F)=(1,160.89)$, with the unchanged 100/150 solver.
The selected ridges coincide, so the same-ridge contrast is the primary
contrast, not another independent finding or additional replay arm.

Development finished at 2026-10-24 09:49 UTC after about 6.54 aggregate active
hours including acceptance. All 80 unique development trajectories and 48
audits are sealed; development storage is about 478 MB. The immutable fresh
confirmation ledger has 12 replicas, both streams, and two recipes: 48
trajectories total. Phase 7D then started automatically. Based on the selected
recipes' measured development timings (about 108 seconds replay and 211 seconds
FD per mixed trajectory), the operational remaining forecast at transition
is approximately 2.5--3.5 active hours including initialization/reporting.
That was the transition forecast, not a change to the immutable historical
after-7A timing artifact. Confirmation took about 2.89 active hours. The final
results above supersede this historical transition snapshot.

The ordinary-policy forecast is 14--24 active compute hours (central estimate
18--20), excluding roughly 3--6 hours of implementation. A proposed 26-hour
aggregate allowance is authorized. One conditional common-ceiling repair
could raise the forecast to 20--40 hours and requires a measured reforecast
within whatever allocation is actually granted. Do not treat per-root ceilings
as independent copies of the overall allowance.

Use the detailed selection, audit, and versioning rules in the plan, not ad hoc
optimizer/regularizer tuning. In particular: reconstruct diagnostic objectives
with their original pre-arrival STM anchors; branches must not alter the live
learner; explicitly assign new seed families to development/confirmation test
partitions; preserve completed Phase 5 evidence; and add the common-ridge replay
control when the final selected FD and replay ridge settings differ. The proposed
Phase 7 entry point is `mnist_experiment.fisher_memory.regularization`.
It runs the complete study autonomously, persists every policy/selection
decision, and checkpoints accepted arrivals and pending objective audits.

The completed production command is retained below for reproducibility. There
is no need to launch it again. From the repository root using the user's venv:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.regularization \
  --root cache/mnist_experiment/plan15/regularization_development_v1 \
  --confirmation-root cache/mnist_experiment/plan15/regularization_confirmation_v1 \
  --allocation-hours 26
```

After any interruption, use the same command with `--resume`. Add `--clear-stop`
only after an intentional STOP request. The 26-hour limit is cumulative across
both roots, previous invocations, and any superseded Phase 7 repair roots;
it is not 26 additional hours per resume. The acceptance compute record also
counts toward that allowance. CUDA access requires ordinary host execution;
the filesystem sandbox hides GPU device files, while PyTorch in the user's
venv detects the RTX 4070 outside it. Do not substitute CPU for production.

The active root's `status.json` and the concise Phase 7 notebook section show
progress. Use `fisher_memory.control stop --root <active-root>` for a cooperative
pause; SIGTERM/Ctrl-C are also cooperative. A hard kill replays at most an
uncommitted arrival or unfinished short audit. Completed units stay immutable.

## Rare Rotated-Domain Confirmation (Phase 5C completed)

The user authorized Phase 5C as written on 2026-10-23, with twelve fresh
four-arm replicas and the predeclared ten-active-hour cumulative cap. The
implementation reuses the Phase 5B trajectory mathematics and fixes FD strength
at its sealed selection, 160.89. Fresh `drip_confirmation` seeds and the
confirmation half of the per-class test-image pool exclude development evidence.
Numerical learning sources are checked against development; only scheduler and
trajectory metadata handling changed. The main notebook receives a concise
live Phase 5C section after the earlier phases. Final inference requires all 48 sealed
trajectories; partial results remain descriptive.

Completed locally on CUDA at 2026-10-23 22:23 UTC after starting at 17:48 UTC.
All 48 trajectories and 62 artifact units are complete and passed full
content-hash validation. The sealed result reproduced exactly and the source
hash still matches. Active runtime was 4.59 hours; artifacts occupy about 284 MB.
No experiment process remains running. Before launch, 244 focused unit/CPU
pipeline tests passed (four expected CUDA-only skips), along with a CUDA
pipeline test. Both CPU and CUDA drills proved exact interrupted/uninterrupted
trajectory equality and unchanged completion seals on resume. Scientific source
hash: `2013757a582afdcceddb36979c77b7c150365117f34f6fde30b8366e53495adb`.
Preserve this source identity for resumption; use a new identity/root for any
scientific correction. Execution used the foreground local session, not `tmux`.

The primary FD gain was +0.4619 NLL-AUC, paired bootstrap 95% interval
[+0.3472, +0.5783], one-sided sign-flip p=0.00024414, with 12/12 favorable
replicas. Mean new-domain acquisition NLL was 2.1161 for replay/EWC versus 1.6542
for FD. Base-retention harm was -0.1566 [-0.1898, -0.1238]; the secondary
interaction was +0.1027 [+0.0512, +0.1585]. The mixed-stream runtime ratio was
1.77 (3.88 versus 6.88 online minutes). Final rotated accuracy averaged 61.98%
versus 65.62%, descriptively. There were zero recorded solver failures/nonfinite
trajectories and 8,318 disclosed under-solved updates. These results establish a
predictive advantage at the tested data/replay budget, not equal-compute
superiority or a memory-transfer mechanism.

The current notebook view is **Phase 5C: Sample and compute efficiency**, at
the bottom after Phase 5B. Results are ordered oldest to newest, and presentation
refreshes preserve that order. Its four-panel figure shows new-domain NLL versus
novel observations, the running acquisition score $A_m^R(n)$, the twelve paired
gains and mean/95% interval, and measured online compute. The score is explicitly
defined as average NLL after novel images 1 through $n$, excluding initialization.
The final means are 2.116 (replay/EWC) and 1.654 (FD), a descriptive 21.8% lower
acquisition loss, not an estimate of the percentage of samples saved. Dashed
lines show the frozen starting model (mean NLL 1.607); a relative efficiency
gain over replay need not be net improvement over initialization throughout
learning. Detailed tables are collapsed, accuracy curves
remain visible, and the older section is labeled as an earlier-study audit.
This presentation lives in reporting-only `report.py`; the frozen analysis and
completed scientific artifacts are unchanged. Refresh the view without training:

```bash
~/.venv/bin/python -m mnist_experiment.fisher_memory.report \
  --root cache/mnist_experiment/plan15/drip_confirmation_v1 \
  --notebook mnist_experiment/fisher_memory_results.ipynb --phase5c
```

The standalone figure is `reports/phase5c_efficiency.png` inside that run root.

The notebook's **Evaluation protocol and comparable NLLs** subsection separates
training batches from evaluation. All twelve sealed data plans were checked:
each has 1,000 unique test IDs (100 per digit by construction), disjoint from
the Phase 5B tuning panels. The four arms share assets and initial rotated NLL
within each replica. All NLLs use the same fixed 30-degree rotated panel and
training-frequency class weights, with no EWC/Ridge/KL penalty included.
The frozen reference is the initial *rotated test* loss (1.6068), not unrotated
training loss; FD's acquisition score (1.6542) averages checkpoints 1--100,
whereas its final NLL (1.5596) uses only checkpoint 100. The definitions and
equations are included in an expandable note. Batch-aligned plotting is labeled
explicitly. This is a reporting clarification, not a scientific change or rerun.

Command used from the repository root (the run is already complete):

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.drip_confirmation \
  --source-root cache/mnist_experiment/plan15/drip_development_v1 \
  --root cache/mnist_experiment/plan15/drip_confirmation_v1 \
  --allocation-hours 10
```

Resume with `--resume`; add `--clear-stop` after a deliberate STOP request.
The run root's `status.json`, frozen `drip-confirmation-ledger-v1`, accepted
batch checkpoints, and final `drip-confirmation-results-v1` are authoritative.
The initial forecast remains 5--8 hours, with approximately 4.4 hours of online
trajectory work projected from the paired development timings before fresh
initialization, checkpoint/report writes, and validation. No additional condition
or replica is authorized by the Phase 6 guiding questions.

## Rare Rotated-Domain Drip (Phase 5B completed)

Phase 5B development of finite-difference KL sample efficiency in a rare
30-degree rotated domain completed on 2026-10-23. Phase 5C fresh confirmation
completed separately above. The pre-run compute forecast was 5--7 hours for 5B and
5--8 hours for 5C; Phase 5B used 5.21 active hours.
Provisional Phase 6 questions are recorded in `plan15.md`: measured FD overhead,
equal-compute replay comparison, intermittent FD/replay allocation, then limited
buffer/budget sensitivity. These notes do not change Phase 5C or allocate Phase 6
compute. No Phase 6 implementation or run has started.
The Phase 5B runner is `fisher_memory.drip`; its deterministic 100-event stream,
four paired arms, strength selection, immutable artifacts, and live section in
`fisher_memory_results.ipynb` are implemented. The accepted-step cached-MNIST
stop/resume smoke passed, as did 220 focused unit tests (four CUDA skips).

Run from the repository root inside `tmux` with the cumulative eight-hour cap:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.drip \
  --root cache/mnist_experiment/plan15/drip_development_v1 \
  --allocation-hours 8
```

Use the same command plus `--resume` after an unplanned interruption. A requested
cooperative stop is a `STOP` file in that root; resume afterward with both
`--resume --clear-stop`. The root's `status.json`, sealed selection unit, and
completed trajectory units are authoritative. A paused invocation exits 3;
neither pause nor failure should trigger a Phase 5C launch automatically.

Launched in `tmux` session `plan15-drip-5b` on 2026-10-22 and resumed in the
foreground after the node was repurposed. The original console output is in
`cache/mnist_experiment/plan15/drip_development_v1.log`. All 46 trajectories,
including four predeclared upper-boundary probes, and all 55 artifact units
passed full content-hash validation. The sealed pilot selected FD strength
160.89 at the new upper grid boundary. Across the four screen replicas, its
mixed-stream acquisition gain was +0.500 mean 30-degree NLL-AUC over replay/EWC,
but the base-only gain was +0.413 without rotated training images. The
descriptive interaction was +0.087 (3/4 positive), and the mean 0-degree AUC
difference in the mixed stream was -0.128 (improvement). Thus most treatment
lift is not specific to exposure to the rare domain; protection or general
regularization remains a strong explanation. These are selected-on-pilot
development estimates, not a significance test or proof of STM-to-LTM transfer.
See the concise main notebook and sealed selection artifact for the development
readout; fresh confirmation is reported above.

## Sigmoid JVP Repair (completed)

The versioned `fisher_memory.jvp_repair` runner completed a fresh
sigmoid-rotation strength reassessment. Its frozen grid included the exact
finite-difference-selected coefficient, approximately 26.8897, three weaker
settings, and one stronger setting. Three fresh pilot replicas selected a JVP
strength subject to the stated retention margin; a disjoint 12-replica cohort
confirmed it. Each replica had a new paired replay/EWC control. The original
completed roots remain read-only.
See Phase 5A in `plan15.md` for the scientific rules.

The code and tiny cached-MNIST pause/resume pipeline were tested. The repair
completed all 42 trajectories in 2.92 active hours; all 59 units passed full
artifact validation. The pilot selected 26.8897, and the 12 fresh confirmation
replicas favored JVP over replay/EWC on current and historical sigmoid NLL AUC
in all 12 pairs. The mean gains were 0.1145 and 0.1359, respectively. The
command used to launch the completed root from the repository root was:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.jvp_repair \
  --source-root cache/mnist_experiment/plan15/production_v4 \
  --root cache/mnist_experiment/plan15/jvp_repair_v1 \
  --allocation-hours 6
```

For an interrupted run, use the identical command plus `--resume`. A paused
invocation exits with status 3, so `&&` will not silently start a later command.
The live notebook is `mnist_experiment/fisher_memory_jvp_repair_results.ipynb`.
Measured prior sigmoid medians were about 299 seconds per JVP trajectory and
114 seconds per replay/EWC control. The pre-run forecast was 3.5--5 hours wall
clock including assets and validation; realized active time was 2.92 hours.

## Optional Method Extension (completed)

The `fisher_memory.extension` entry point now supports separately sealed JVP
and/or directional-KL Phase 5 trajectories with explicit environment scope,
replica count, cumulative compute allocation, and `--resume`. It reuses the
original `production_v4` replay/EWC summaries as verified read-only paired
controls, freezes pilot-selected strengths, and reproduces the confirmation
seed family, asset-plan hashes, and exact initial learner-state fingerprints in a
new run root. The extension notebook is published separately and reports
progress plus paired current/historical NLL AUC. See the Phase 5 command in
`plan15.md`.

**Completed on 2026-10-19.** `cache/mnist_experiment/plan15/extension_v1`
contains all 72 JVP and directional-KL trajectories (12 paired replicas for
each of three environments and two methods). Its 97 artifact units passed full
content-hash validation; no trajectories are missing or unsealed. The run used
approximately 4.61 cumulative active hours. The main results notebook now
includes a concise Phase 5 comparison, and the separate extension notebook
retains the progress and per-replica view.

Relative to paired replay/EWC, both alternative penalties show positive mean
current and historical NLL-AUC gains in rotation. Directional KL is closest to
the selected finite-difference method at lower measured online cost; JVP is
slower and substantially weaker on sigmoid rotation. Neither alternative shows
a clear digit-9 mixture gain. These are secondary nominal paired comparisons,
not a new familywise-confirmatory claim or proof of STM-to-LTM transfer.

Verification: 216 focused tests passed with four expected CUDA-only skips,
including a cached-MNIST CPU entry-point test that interrupted after one
trajectory and resumed the remaining method. The real sealed source root passed
the extension-ledger preflight for all 72 method/environment/replica jobs before
the extension was launched.

This added scientific source changes the current source hash, so do not attempt
to resume the completed `production_v4` study using the new source tree. Its
sealed artifacts remain the read-only reference, and a reporting-only refresh
does not require resuming its scientific scheduler.

## Status

**Phases 1--4 completed on 2026-10-18 (America/Los_Angeles).**
`production_v4` contains 284 completed trajectories: 140 development and 144
fresh confirmation trajectories (12 replicas x 3 environments x 4 conditions).
All 630 artifact units passed a full content-hash validation. There are no
incomplete units, recorded solver failures, nonfinite results, or missing
primary retention endpoints. Phase 5 was run subsequently, as noted above.
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
