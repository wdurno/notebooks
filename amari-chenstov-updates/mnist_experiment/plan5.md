# Implementation Plan 5: Rotated MNIST Transfer and Revisitation

Plan 5 is a deliberately detachable follow-up to the completed experiments in
[plan3.md](plan3.md) and [plan4.md](plan4.md). It asks two related questions
that the digit-9 mixture path could not settle:

> Can a compact EWC summary make low-data learning statistically more
> efficient when the new task reuses structure learned by a high-quality
> starter model?

> Can an online Fisher-risk recommendation become useful when the learner
> traverses, reverses, and revisits a visibly nonlinear environmental path?

The treatment is deterministic image rotation. The initial model learns
upright MNIST, and the online environment follows two forward passages through
the same rotation family, with one return passage between them:

$$
0^\circ\longrightarrow15^\circ\longrightarrow30^\circ
\longrightarrow0^\circ\longrightarrow15^\circ\longrightarrow30^\circ.
$$

This schedule supplies first exposure, return, and repeated exposure without
inventing unrelated tasks. It may nevertheless fail to create useful transfer
or controller variation. Every phase therefore has an explicit stop decision,
and no Plan 5 result is required by the accepted Plans 1--4 baseline.

## Motivation

The completed MNIST experiments support a practical but limited result: EWC
preserves old capabilities under low-data adaptation, especially when paired
with a small replay buffer. They do not show that EWC reduces the observations
needed to learn digit 9. The original 512-parameter CNN and the appearance of a
previously absent class may offer too little reusable task structure for that
question.

Rotated MNIST creates a more direct transfer problem. Labels and semantic
content remain fixed while the observation map changes. A high-quality upright
classifier already contains useful digit features, and the online learner must
adapt those features using only a few rotated observations. Current-only,
replay, EWC, and Hybrid learners can then be compared by how quickly they
recover current-angle performance and how well they retain performance at
previously encountered angles.

The repeated path also creates a stronger adaptive-composition challenge than
the monotone digit-9 mixture. Let $\varphi$ denote rotation angle and

$$
\gamma(\varphi)=\theta^\star(\varphi).
$$

Even at constant angular speed, the induced parameter speed and local Fisher
quadratic need not be constant:

$$
\frac{d\theta^\star}{dt}
=
\gamma'(\varphi_t)\frac{d\varphi_t}{dt},
\qquad
\|d\theta_t\|_{\mathcal I_t}^2
=d\theta_t^T\mathcal I_t d\theta_t.
$$

The return from $30^\circ$ to $0^\circ$ reverses the environmental direction,
and the second ascent revisits a previously observed path. These events may
separate a useful adaptive recommendation from a noisy response to generic
movement. They do not guarantee that EDR will outperform the best fixed
$\pi$ found after repeated research trials. In the target setting, however,
that post facto fixed policy is generally unavailable; approaching it from one
online trajectory may itself have applied value.

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Scientific, modularity, and schedule contracts | Complete |
| 1 | Detachable rotation infrastructure | Complete |
| 2 | Learnability and geometric feasibility audit | Complete |
| 3 | Low-data transfer pilot | Complete |
| 4 | Repeated-path memory and replay comparison | Complete |
| 5 | EDR recommendation and closed-loop challenge | Complete (single-lap retry stopped) |
| 6 | Independent confirmation of promoted contrasts | Pending |
| 7 | Findings integration or clean retirement | Pending |

## Isolation Contract

Plan 5 must remain removable without changing any accepted result, immutable
artifact, or execution path from Plans 1--4.

1. Put Plan 5 implementation under
   `mnist_experiment/rotated_mnist/`, tests under
   `test/unit/rotated_mnist/`, and generated artifacts under
   `cache/mnist_experiment/rotated_mnist/`.
2. Use Plan 5-specific configuration, stream, metric, and artifact schema
   names. Never reinterpret a historical field such as `p`, `pi`, or
   `schedule_kind` to mean rotation angle.
3. Existing entry points may be imported as libraries, but they must not
   import the rotated module. Prefer a thin Plan 5 runner and adapters over
   conditionals spread through `run_controller.py`.
4. A narrowly useful shared helper may move into `src/` only when its behavior
   is already covered by tests and the move leaves legacy outputs unchanged.
   Rotation-specific schedules, transforms, metrics, and orchestration stay in
   the detachable module.
5. Completed Plan 5 runs are immutable under the repository artifact contract.
   Removing source code never authorizes deleting or rewriting their stored
   provenance.
6. Do not revise [mathematical_overview.ipynb](../mathematical_overview.ipynb),
   the accepted deployment recommendation, or
   [mnist-findings.ipynb](../mnist-findings.ipynb) until a phase gate identifies
   evidence worth integrating.

The practical discard operation should consist of removing the two
Plan 5 source/test directories, its small configurations and notebook, and
references from future documentation. No migration of Plans 1--4 should be
required.

## Frozen Handoff

Unless a phase check-in explicitly reopens one item, retain:

- the canonical 512-parameter CNN and complete trainable-network Fisher;
- ordinary cross-entropy as the negative log likelihood;
- fresh high-quality initialization for every independent replica;
- $m=8$ current observations per environmental update;
- 50 L-BFGS inner iterations per learner update;
- rank-8-plus-diagonal Fisher summaries;
- direct EMA Fisher updating with no LFU or HVP calculation;
- fixed $\pi=.05$ as the predeclared practical EWC baseline;
- replay sampling with replacement and Plan 3 clean archive recursion;
- Replay B32 and Hybrid B32 as the bounded-memory applied conditions;
- complete paired trajectories and immutable append-only replicas; and
- pre-update evaluation and exposure semantics from Plan 3.

The canonical model is frozen initially to isolate the environmental change.
If it cannot attain the Phase 2 high-sample learnability threshold at
$30^\circ$, stop for a model-capacity check-in. A larger model must be a new,
versioned treatment and must not silently replace the canonical network.

## Experimental Contracts

### Statistical process

Let $(X_i,Y_i)$ be a canonical MNIST observation and let
$T_\varphi(X_i)$ be its deterministic rotation by $\varphi$ degrees. The
online observation is

$$
X_{i,t}^{(\varphi)}=T_{\varphi_t}(X_i),
\qquad Y_{i,t}^{(\varphi)}=Y_i.
$$

Use the fixed empirical ten-digit MNIST class distribution throughout Plan 5.
There is no changing digit-9 mixture and no Plan 5 variable named $p_t$.
Rotation is the sole environmental coordinate:

$$
\varphi
\longmapsto
\theta^\star(\varphi)
\longmapsto
\mathcal I(\theta^\star(\varphi)).
$$

This remains the unique likelihood Fisher induced by the model loss. Do not
introduce a second Fisher estimand $\mathcal I(\theta,\varphi)$ or treat angle
as an additional model parameter. The original process updates the network
parameters; the auxiliary process maintains the Fisher summary along that
realized parameter path.

The high-quality initializer is fit on upright observations from all ten
classes. Its data are disjoint from online and evaluation streams under the
existing sampler contracts. Every paired treatment starts from the same model
and initial Fisher within a replica.

### Rotation transform

Use one deterministic, versioned transform:

- bilinear interpolation;
- fixed $28\times28$ output without expansion;
- zero fill in normalized background coordinates;
- no random crop, translation, reflection, antialias toggle, or secondary
  augmentation; and
- floating-point transformation before model normalization.

Record the library version, interpolation mode, fill value, tensor dtype,
angle, source observation identifier, and transformed-content hash. A change
to any transform setting creates a new stream schema and cannot resume an old
run.

Rotation must be applied identically to training and current-angle evaluation
observations. Keep one unrotated holdout and one fixed angle-panel holdout for
retention measurements. Do not materialize all rotated MNIST variants in
source control; deterministic cache entries belong under `cache/`.

### Repeated schedule

Let the frozen knot sequence be

$$
(a_0,a_1,a_2,a_3,a_4,a_5)
=
(0,15,30,0,15,30)\text{ degrees}.
$$

Use $L=20$ transitions per leg. After sharing adjacent endpoints, the resolved
trajectory has

$$
K=5L+1=101
$$

environmental points and 100 transitions. For leg $j=0,\ldots,4$ and
$q=0,\ldots,L$,

$$
\varphi_{jL+q}
=a_j+\frac{q}{L}(a_{j+1}-a_j),
$$

with shared knot observations represented once. This gives:

- first ascent: $0^\circ\to15^\circ\to30^\circ$;
- return: $30^\circ\to0^\circ$; and
- second ascent: $0^\circ\to15^\circ\to30^\circ$.

Store every resolved angle, leg identifier, direction, knot flag, cumulative
angular distance, and schedule hash. Do not reconstruct the path from labels
inside a results notebook.

The fixed 20-transition-per-arrow contract implies angular increments of
$0.75^\circ$ on each $15^\circ$ forward leg and $1.5^\circ$ on the
$30^\circ\to0^\circ$ return. The return is therefore a combined direction and
speed challenge. Do not attribute its controller response to path geometry
alone. Matched first and second ascents retain equal angular speed and remain
the clean revisitation comparison.

The main comparison is between matched locations on the first and second
ascents. The return leg is scientifically meaningful, not a burn-in or hidden
reset. Parameters, Fisher summaries, replay buffers, EDR state, and exposure
counts continue through every knot. No learner is reset at $30^\circ$ or at
the second $0^\circ$.

### Pairing and nested data

Within a replica, every condition receives the same canonical observation
identifiers, labels, angles, transformed tensors, evaluation panels, and
non-treatment seeds. Rotation does not authorize resampling a more convenient
class sequence.

Preserve nested low-data streams so a later $m\in\{1,2,4,8\}$ sensitivity can
reuse prefixes of the same eight observations at each step. The principal
Plan 5 treatment remains $m=8$ until Phase 3 establishes that the task is too
easy or too difficult at that level. Changing $m$ creates a distinct run, not
a notebook-side subsample.

### Treatment hierarchy

Do not launch the full treatment family at once.

1. **Learnability controls:** frozen initializer, current-only adaptation, and
   a high-data current-angle fit used only to establish attainable performance.
2. **EWC isolation:** current-only versus direct-EMA fixed-$\pi=.05$ EWC from
   the same starter model and stream.
3. **Memory comparison:** current-only, EWC, Replay B32, Hybrid B32, and fresh
   unbounded replay. The unbounded condition is a resource-unconstrained
   control, not the deployment recommendation.
4. **Adaptive challenge:** prospective fixed $\pi=.05$, a small predeclared
   research-only fixed bracket
   $\pi\in\{.01,.025,.05,.10\}$, and EDR initialized at $.05$. Add Hybrid EDR
   only after EWC-only actuation passes its gate.

Use separate, uncluttered contrasts rather than one five-condition plot. The
frozen initializer is context for adaptation and forgetting; it is not a
learning competitor.

### EDR contract

Retain Plan 4's Fisher-risk coefficient estimand:

$$
\widehat R_t(\pi)
=(1-\pi)^2A_t+\pi^2B_t,
$$

$$
\bar A_t=(1-\gamma_\pi)\bar A_{t-1}+\gamma_\pi A_t,
\qquad
\bar B_t=(1-\gamma_\pi)\bar B_{t-1}+\gamma_\pi B_t,
$$

$$
\pi_t^{\mathrm{EDR}}
=\operatorname{clip}
\left(
\frac{\bar A_t}{\bar A_t+\bar B_t},
\pi_{\min},\pi_{\max}
\right).
$$

Use $H_\pi=4$ accepted updates, $\pi_{\min}=.01$,
$\pi_{\max}=.95$, and a $.05$ cold-start action as the predeclared primary
controller. The action remains predictable and one realized $\pi_t$ controls
the EWC objective, Fisher EMA, archive consolidation, and effective-information
recursion.

Replace Plan 4's prevalence-distance trend half-life with an explicitly named
angular-distance half-life $H_\varphi=7.5^\circ$, one quarter of the
$0^\circ$--$30^\circ$ range. For transition distance
$\Delta s_t=|\varphi_t-\varphi_{t-1}|$, use

$$
\gamma_t=1-2^{-\Delta s_t/H_\varphi}.
$$

The absolute distance is deliberate: direction belongs in the vector trend,
while elapsed environmental distance controls forgetting. The return leg must
not produce a negative or explosive EMA gain. Sensitivities to
$H_\varphi\in\{3.75^\circ,15^\circ\}$ are assumption checks and cannot replace
the primary value using predictive outcomes.

EDR is an unconfirmed recommendation mechanism. Movement of its action is not
success, and fixed $.025$ is not a fair prospective baseline because earlier
adaptive work helped identify it. The best result in the predeclared fixed
bracket is a post facto research benchmark: obtaining it requires the repeated
trials that the intended online application may not permit. Judge prospective
utility first against fixed $.05$, then report how closely EDR approaches the
best tested fixed policy without task-specific tuning.

### Outcomes

Preserve complete per-replica trajectories. The principal predictive outcomes
are:

- current-angle environmental multiclass accuracy and NLL as the primary
  transfer-sensitive outcomes under the fixed empirical ten-digit
  distribution;
- per-class precision and recall for all ten digits, together with worst-class
  recall, as localization guards against an aggregate gain hiding class
  failure;
- environmental multiclass accuracy, NLL, and per-class precision and recall
  reported separately at each angle in the fixed
  $\{0^\circ,15^\circ,30^\circ\}$ panel, never pooled into one unspecified
  `accuracy`;
- upright environmental multiclass accuracy, NLL, and per-class precision and
  recall as retention outcomes;
- digit-9 OvR accuracy, precision, and recall as supplemental continuity
  metrics only; digit 9 has no privileged estimand in Plan 5;
- ECE and Brier score;
- cumulative current observations and optimizer evaluations;
- learner time, GPU time, persistent bytes, and peak memory; and
- replay occupancy, Fisher-summary bytes, and total stored observation bytes.

For the first ascent, define environmental-accuracy AUC against cumulative
current observations as

$$
\operatorname{AUC}_{\mathrm{env},\uparrow1}
=
\frac{1}{n_{\uparrow1}}
\int_0^{n_{\uparrow1}}
A_{\mathrm{env}}^{\mathrm{current}}(n)\,dn,
$$

and define the corresponding integrated NLL

$$
\operatorname{AUNLL}_{\mathrm{env},\uparrow1}
=
\frac{1}{n_{\uparrow1}}
\int_0^{n_{\uparrow1}}
\operatorname{NLL}_{\mathrm{env}}^{\mathrm{current}}(n)\,dn.
$$

Calculate both by the trapezoidal rule over stored evaluation points. Higher
accuracy AUC and lower NLL AUC indicate more efficient adaptation along the
same paired angle and observation schedule. Report per-class trajectories and
worst-class recall beside these aggregate summaries rather than constructing a
balanced aggregate statistic.

For the adaptive comparison, let

$$
Y_0
=\operatorname{AUC}_{\mathrm{env},\uparrow1}^{\pi=.05},
\qquad
Y_\star
=\max_{\pi\in\{.01,.025,.05,.10\}}
\operatorname{AUC}_{\mathrm{env},\uparrow1}^{\pi},
$$

and let $Y_{\mathrm{EDR}}$ be the corresponding EDR result. Define the
available fixed-policy tuning opportunity and the fraction captured online as

$$
\Delta_\star=Y_\star-Y_0,
\qquad
\rho_{\mathrm{EDR}}
=\frac{Y_{\mathrm{EDR}}-Y_0}{\Delta_\star}.
$$

If $\Delta_\star<.005$, classify the trajectory as having no practically
meaningful fixed-policy tuning opportunity and leave
$\rho_{\mathrm{EDR}}$ undefined. Otherwise, automatic selection requires
$\rho_{\mathrm{EDR}}\geq.80$ and
$Y_\star-Y_{\mathrm{EDR}}\leq.01$, together with no material NLL, per-class
precision or recall, retention, calibration, or resource regression. Report
those guardrails separately rather than combining them into one score.

Define first-pass transfer efficiency through complete trajectories, not a
single noisy crossing. Report each mean paired trajectory, its 95% Student-$t$
interval, the two integrated summaries, and an observation-efficiency curve
relative to current-only. Here the two named summaries are environmental
accuracy AUC and environmental NLL AUC. Threshold-crossing summaries are
secondary and require
persistence across at least three evaluation points.

For the repeated path, compare matched ascending angles. Let
$Y_{\uparrow1}(\varphi)$ and $Y_{\uparrow2}(\varphi)$ denote first- and
second-ascent metrics. Report

$$
\Delta_{\mathrm{revisit}}Y(\varphi)
=Y_{\uparrow2}(\varphi)-Y_{\uparrow1}(\varphi),
$$

along with area under each ascent, return-leg recovery, and performance at all
knots. A positive revisit difference may reflect memory, additional total
training, or both; treatment contrasts are required before interpreting it as
EWC-specific memory.

For EDR, retain Plan 4's single-trajectory diagnostics: prequential
calibration ratio $C_t$, recommendation hysteresis, total variation, bound
occupancy, tail settling, estimated Fisher-risk opportunity, and divergence
between internal opportunity and realized predictive gain. Add alignment with
angle, angular direction, knot proximity, and first-versus-second exposure.

### Decision language

Use the following interpretations:

- **Transfer success:** EWC materially improves environmental-accuracy AUC,
  environmental NLL AUC, or persistent observation efficiency over
  current-only while preserving upright capability. Per-class precision,
  recall, and worst-class recall must remain coherent so aggregate accuracy
  cannot hide a localized failure.
- **Memory success:** EWC or Hybrid improves matched-angle second-pass or
  return performance over its no-EWC comparator without merely refusing to
  adapt.
- **Dynamic adaptive success:** EDR provides predictable, non-floor-driven
  action and improves different path regions beyond every tested fixed policy
  without material task, calibration, retention, or compute regression.
- **Automatic-selection success:** starting from $.05$, EDR approaches the
  post facto best tested fixed policy within a predeclared practical margin,
  without task-specific trials or a material secondary regression. It need not
  beat that research-only comparator to be useful.
- **Prospective adaptive value:** EDR improves over untuned fixed $.05$ and
  remains acceptably close to the best tested fixed policy. This is weaker than
  evidence of optimal stochastic control but directly addresses a deployment
  where repeated policy tuning is unavailable.
- **Diagnostic only:** the Fisher-risk recommendation varies coherently or
  calibrates toward a useful operating region but its closed-loop action does
  not improve predictive outcomes.
- **Stop:** the canonical model cannot learn the transformed task, EWC does
  not improve low-data transfer, or apparent adaptive value is explained by
  clipping, leakage, extra observations, or post-event reaction. EDR is also
  practically unhelpful when it materially underperforms both the prospective
  $.05$ baseline and the best tested fixed bracket without a compensating
  resource advantage.

## Phase 0: Scientific, Modularity, and Schedule Contracts

### Goal

Freeze what rotated MNIST estimates, how it can be removed, and how the
two-pass path is represented before implementation.

### Scope

1. Confirm that angle replaces prevalence as the sole environmental coordinate
   in this adjunct experiment. Keep the empirical ten-digit class distribution
   fixed.
2. Freeze the deterministic transform and the six-knot, 101-point schedule.
3. Freeze initialization, pairing, no-reset semantics, evaluation panels, and
   first-versus-second ascent matching.
4. Define the source, test, configuration, artifact, and notebook boundaries
   required for clean removal.
5. Freeze learnability, transfer, memory, graded adaptive, diagnostic-only,
   and stop language before predictive results exist. Predeclare the practical
   margin used to decide whether EDR approaches the best tested fixed policy.
6. Audit the proposed module against the immutable artifact contract and prove
   that no legacy configuration hash or import path changes.

### Verification

- A deterministic schedule audit proves exact knots, 20 transitions per arrow,
  $K=101$, and no hidden reset or duplicate update. Phase 1 promotes this audit
  into unit tests against the implemented schedule module.
- A transform prototype audit proves repeated deterministic output and fixed
  shape under the frozen environment. Phase 1 adds versioned implementation
  tests without treating one platform-specific byte hash as a mathematical
  invariant.
- A legacy import and configuration-hash audit passes before the rotated module
  exists. Phase 1 repeats it after implementation.
- The statistical contract contains one likelihood Fisher and keeps original
  and auxiliary processes distinct.

### Check-in

Approve the experiment's estimand, repeated schedule, and discard boundary
before adding executable code.

### Completion Record

**Status:** Complete (2026-08-29)

- Froze rotation angle $\varphi_t$ as the sole changing environmental
  coordinate while retaining the empirical ten-digit class distribution and
  the unique likelihood Fisher. The upright initializer learns all ten digits;
  original and auxiliary processes retain their existing meanings.
- Froze deterministic bilinear rotation, fixed $28\times28$ output, zero fill,
  no expansion, and no secondary augmentation. A float32 prototype under
  PyTorch `2.10.0+cu128` and torchvision `0.25.0+cu128` produced bit-identical
  repeated output with shape `(1, 28, 28)`; its diagnostic SHA-256 was
  `cbb2f72b8a8d815b1a3715076bce5d238de15e6e9c18b9d3a1cae312876e24f8`.
  This records the audit environment and is not a cross-version scientific
  invariant.
- Verified the six approved knots at indices
  $\{0,20,40,60,80,100\}$, yielding $K=101$ points and 100 transitions with
  no reset. The audit made the schedule's speed structure explicit: forward
  $15^\circ$ legs move by $.75^\circ$ per update and the return moves by
  $1.5^\circ$ per update. Return-leg effects therefore mix direction and speed;
  matched ascents remain the clean repeated-path comparison.
- Froze paired initialization, source observations, transformed tensors,
  evaluation panels, no-reset state, and first-versus-second ascent matching.
- The initial Phase 0 review froze digit-9 OvR AUC as the primary transfer
  metric. Before any low-data treatment run, the Phase 2 check-in exposed that
  this inherited the old emerging-nine estimand even though Plan 5 keeps the
  empirical ten-digit distribution fixed. The amended primary outcomes are
  environmental-accuracy and NLL AUC, with per-class precision, recall, and
  worst-class recall as localization guards. Digit-9 metrics remain
  supplemental continuity fields only.
- Added the accepted adaptive opportunity contract. An improvement below
  `.005` leaves no meaningful fixed-policy tuning opportunity. Otherwise EDR
  automatic selection requires at least 80% opportunity capture, an absolute
  gap no larger than `.01` from the best tested fixed policy, and no material
  guardrail regression.
- Froze the detachable source, test, configuration, notebook, and cache
  namespaces. Existing runners will not import the Plan 5 module, and no
  historical field is repurposed for angle.
- The legacy configuration, Plan 3, Plan 4, and EDR contract tests completed
  with `54 passed`. A direct import audit of the principal initialization,
  controller, command-center, configuration, Plan 3, Plan 4 analysis, EDR,
  and results-analysis modules also passed. Since Phase 0 changes no executable
  code, legacy configuration hashes remain unchanged by construction. The
  complete unit suite subsequently passed with `318 passed, 2 skipped`.

**Gate recommendation:** proceed to Phase 1. Implement only the detachable
rotation pipeline and its CPU smoke; do not begin learnability fits or learner
treatment comparisons until the Phase 1 boundary and artifact checks pass.

## Phase 1: Detachable Rotation Infrastructure

### Goal

Implement the smallest self-contained pipeline capable of producing and
reading one immutable rotated-MNIST smoke trajectory.

### Scope

1. Add `mnist_experiment/rotated_mnist/` with focused modules for transform,
   schedule, stream materialization, configuration, execution, artifact
   validation, and lightweight analysis.
2. Reuse stable initialization, model, Fisher representation, optimizer, and
   metric helpers through imports. Wrap legacy runner behavior rather than
   adding a rotation branch throughout it.
3. Add stable component seeds for source sampling, transformation/cache,
   online order, evaluation panels, optimizer, and randomized numerical
   routines.
4. Store resolved angles and transformed observation identities in a
   Plan 5-specific stream artifact before learner execution.
5. Add a tiny CPU configuration with shortened legs and one condition. It must
   exercise both direction changes and second exposure.
6. Add an artifact-only notebook skeleton that refuses incomplete or
   incompatible runs and performs no training or data download.

### Verification

- Unit tests cover interpolation settings, schedule indexing, pairing,
  component seeds, schema validation, immutable collision behavior, and
  notebook loader failure modes.
- The CPU smoke completes, resumes only an incomplete run, and refuses to
  mutate its `COMPLETED` directory.
- Existing unit tests retain exact behavior.
- No Plan 5 artifact is written outside its cache namespace.

### Check-in

Review the module boundary and one smoke artifact. If rotation required broad
legacy-runner changes, stop and simplify before scientific compute.

### Completion Record

**Status:** Complete (2026-08-29)

- Added the self-contained `mnist_experiment/rotated_mnist/` package with
  strict configuration, repeated-angle schedule, deterministic transform,
  all-digit partitions, nested paired streams, immutable artifacts, a
  current-only runner, and lightweight analysis loaders. Added local agent
  notes that preserve the discard boundary.
- Reused the canonical 512-parameter model, parameter layout, established
  classifier metrics, L-BFGS proposal implementation, dataset loader, and seed
  derivation through imports. No file in `src/` and no legacy runner or
  command center changed or imports the rotated package.
- Added independent named seeds for model initialization, partitioning,
  initialization order, the online stream, references, evaluation panels,
  transform caching, optimization, and numerical randomization. Every seed is
  stored in the immutable manifest.
- Added a master stream width of eight observations per point. An active
  $m\in\{1,2,4,8\}$ run consumes the corresponding prefix of one master draw,
  preserving future nested low-data pairing. Unit evidence verifies that
  separately generated $m=2$ tensors and identifiers exactly equal the first
  two entries of the matched $m=4$ stream.
- Materialized and persisted partitions, resolved angles, leg/direction/knot
  metadata, source identifiers, labels, rotated tensors, per-observation
  hashes, and aggregate stream hashes before initialization or learner
  execution. Rotation operates on the canonical float `ToTensor` output; the
  present model pipeline has no additional normalization.
- Added Plan 5 configuration, metric, artifact, partition, stream, transform,
  and schedule schema version 1. Completed runs use an atomic `.incomplete`
  transaction and `COMPLETED` marker, reject mutation, and preserve incomplete
  state for explicit `--resume`.
- The strict loader validates configuration and manifest identity, partition,
  schedule, stream and transformed-tensor hashes, scalar trajectory ordering,
  parameter/displacement identity, trajectory hash, and initial/final model
  state hashes. Its notebook path reads only lightweight JSON artifacts and
  rejects incomplete or incompatible run roots.
- Added the shortened six-point CPU smoke configuration. Its finalized
  immutable run is
  `rotated_mnist_phase1_smoke__replica-0001__d478561f92b548bb`, with full
  configuration hash
  `d478561f92b548bb1995195a587ee5288bb01b629c5ff6e430c3b2e7d520ba16`,
  stream hash
  `d3bd72a6cadbbc75ade6dda33d0dcf0e98d5053c855a9a85d2d128200c286184`,
  and trajectory hash
  `cc992f16dc4d0bfd1168b2f8efeedaa9cec4ee67b3815fa125c8bfbc3bef1bb8`.
  It completed all five updates in 4.2 seconds and occupies approximately
  160 KiB.
- The smoke fits only one epoch on 512 upright observations and reached
  `10.5%` initializer accuracy. This deliberately cheap result verifies
  plumbing only; it supplies no evidence about rotated-task learnability,
  EWC transfer, replay, Fisher geometry, or EDR.
- Added `rotated_mnist/results.ipynb`. It performs no training, download,
  repair, or resumption, executes from the stored artifacts in 1.92 seconds,
  and labels smoke output as non-scientific.
- The 13 focused Plan 5 unit tests pass. The complete unit suite passes with
  `331 passed, 2 skipped`, and an explicit 11-module legacy import audit
  passes. Completed-run collision and incomplete-resume behavior are covered;
  an actual completed smoke rerun was also refused.

**Gate recommendation:** proceed to Phase 2. The module boundary survived and
the artifact pipeline is operational. Phase 2 may fit high-data references and
audit learnability/geometry, but must not compare EWC, replay, Hybrid, or EDR
closed-loop treatments before its own gate is reviewed.

## Phase 2: Learnability and Geometric Feasibility Audit

### Goal

Establish that the canonical model can learn the rotation family and that the
repeated path creates measurable, non-artifactual changes before comparing
continual-learning policies.

### Scope

1. Evaluate the upright initializer without adaptation on the fixed angle
   panel. This measures zero-shot rotational transfer.
2. Fit independent high-data reference models at
   $\varphi\in\{0,7.5,15,22.5,30\}^\circ$ from matched initialization and
   record attainable current-angle environmental multiclass accuracy, NLL,
   per-class precision and recall, calibration, displacement, and Fisher
   summaries. Retain digit-9 OvR fields only as supplemental continuity
   metrics.
3. Repeat selected endpoint fits to estimate optimization and finite-sample
   variation. Do not interpret one optimizer realization as
   $\theta^\star(\varphi)$.
4. Inspect image examples and classwise confusion at every reference angle to
   detect clipping or interpolation artifacts.
5. Along the reference path, report Euclidean displacement, predictable
   Fisher quadratic displacement, Fisher change, and directional asymmetry.
   Do not invert or pseudoinvert a Fisher estimate.
6. Run the EDR coefficient calculation offline as a design diagnostic only.
   No adaptive learner is authorized in this phase.

### Gate

- **Proceed:** the canonical model reaches acceptable high-data performance at
  $30^\circ$, zero-shot performance leaves room for learning, and path/Fisher
  diagnostics exceed repeated-fit noise over a material interval.
- **Capacity check-in:** the transformed task is learnable in principle but the
  canonical model is the bottleneck. Specify one versioned medium model and
  repeat only this phase.
- **Stop:** rotation is either trivial, dominated by transform artifacts, or
  not learnable enough to support the intended transfer question.

Do not choose model capacity, angle range, or interpolation mode from EWC or
EDR predictive outcomes.

### Verification

- Reference artifacts record optimizer convergence diagnostics and repeated-fit
  variability.
- Geometry summaries use matched parameter ordering and no Fisher inverse.
- The notebook distinguishes zero-shot transfer, high-data attainability, and
  continual-learning performance.

### Check-in

Decide whether the canonical model and $0^\circ$--$30^\circ$ range create a
useful task before running treatment comparisons.

### Execution record

Phase 2 is complete. The detachable package now contains a strict audit
configuration, an immutable high-data reference runner, held-out dense Fisher
estimation, endpoint-refit noise diagnostics, transform examples, classwise
confusion artifacts, and artifact-only notebook views. The focused suite has
21 passing tests; the repository suite has 339 passing tests and 2 skips.

The authoritative completed scientific artifact is
`cache/mnist_experiment/rotated_mnist/audits/rotated_mnist_phase2_learnability_holdout_audit__replica-0001__1a2146c38e8a2490`.
It contains five matched primary fits, two additional independent fits at each
endpoint, 6,000 held-out scores per Fisher estimate, and an 8,000-observation
reporting panel. The complete audit took 36.13 seconds on the RTX 4070.

An earlier immutable artifact ending in `1b47b1a1d1ea2a6a` is retained but
superseded: its initializer target check touched the reporting panel. The
strict-holdout rerun used only the 2,000-observation validation split for that
check. Both stopped after the same first epoch and produced the same model
state and scientific metrics; only the corrected artifact is evidence for the
gate.

The predeclared gate passed all three checks:

- zero-shot environmental accuracy fell from `.857` at $0^\circ$ to `.480`
  at $30^\circ$, leaving a clearly nontrivial adaptation problem;
- the high-data $30^\circ$ reference reached `.889` environmental accuracy
  and `.375` NLL, with classwise recall at or above `.800`; and
- median adjacent-angle relative Fisher change was `.134`, versus `.081` for
  endpoint refits, giving a predeclared Fisher-change signal/noise ratio of
  `1.66`.

The geometric result is real but moderate. Adjacent-angle Fisher quadratics
were approximately `.10`--`.14`, while fixed-angle refit quadratics were
`.032`--`.048`. Directional asymmetry remained small at `.025`--`.062`.
Accordingly, later controller work should not assume dramatic curvature.

Transform inspection found no material artifact: total absolute intensity at
$30^\circ$ was `.9996` times its upright value, and only about `1.0%` of that
mass occupied the two-pixel border. Classwise recall remained at least `.80`
for every digit in the high-data $30^\circ$ fit.

All reference optimizations used the full 12-epoch budget, with selected
epochs 11 or 12. This does not establish exact optimizer convergence. Endpoint
predictive variation was nevertheless small: the three $30^\circ$ fits had
environmental accuracies `.888`, `.888`, and `.892`. As in earlier phases,
these are practical high-data references rather than literal globally solved
MLEs.

The ideal-covariance offline risk diagnostic recommended unclipped local
$\pi$ values from `.0018` to `.0092` for $m=8$. It did not actuate a learner
and supplies no predictive EDR evidence.

**Gate decision:** proceed to Phase 3 with the canonical model and frozen
$0^\circ$--$30^\circ$ range. Rotation is learnable, nontrivial without
adaptation, visually well behaved, and accompanied by Fisher change exceeding
refit noise. Do not begin Phase 3 until its treatment pilot is reviewed at the
next check-in.

## Phase 3: Low-Data Transfer Pilot

### Goal

Test the primary begged question: does the EWC information summary reduce the
data needed to adapt from upright to rotated MNIST?

### Scope

1. Use only the first ascent
   $0^\circ\to15^\circ\to30^\circ$ for the initial mechanism pilot.
2. Compare paired current-only and direct-EMA fixed-$\pi=.05$ EWC learners.
   Include the frozen initializer and high-data reference only as context.
3. Run one development replica at $m=8$. If both learners saturate immediately
   or neither learns, stop before adding conditions and choose at most one
   nested $m$ sensitivity from $\{1,2,4\}$.
4. Measure $\operatorname{AUC}_{\mathrm{env},\uparrow1}$ and
   $\operatorname{AUNLL}_{\mathrm{env},\uparrow1}$ as the primary transfer
   summaries. Report per-class precision and recall, worst-class recall,
   upright retention, angle-specific panel metrics, calibration, expected
   observations to persistent environmental thresholds, optimizer work, time,
   and memory as guardrails. Keep digit-9 OvR fields supplemental.
5. Inspect complete trajectories before reducing them to threshold or endpoint
   tables. Accuracy barely changing is a negative result, not preservation of
   applied value.

### Gate

- **Promote:** EWC shows a material paired low-data advantage over current-only
  in environmental-accuracy AUC, environmental NLL AUC, or persistent
  observation efficiency, with coherent classwise behavior and preserved
  upright performance.
- **Retention only:** EWC preserves upright capability but does not improve
  current-angle environmental learning. Record that result and do not claim
  statistical efficiency from retention alone.
- **Stop:** EWC is neutral or harmful on both adaptation and retention, or the
  model/task gate was misleading.

The development replica selects whether further compute is warranted; it does
not support an inferential claim.

### Verification

- Conditions share initialization, stream, optimizer budget, and evaluation
  panels exactly.
- The EWC treatment differs only through its information summary and resulting
  objective.
- Observation-efficiency calculations use cumulative observations actually
  available before each evaluation.

### Check-in

Review whether Plan 5 has demonstrated transfer, retention only, or no useful
EWC effect before introducing replay or the repeated path.

### Execution record

Phase 3 is complete. A Phase 3-specific paired runner and immutable artifact
schema were added without weakening or reinterpreting the Phase 1 schema. The
runner fits one fresh shared initializer, estimates its upright Fisher from
the complete disjoint 20,000-observation reference partition, compresses that
estimate to rank 8 plus diagonal, and then applies the same 320-observation
rotated stream to both learners. The EWC learner uses the accepted predictable
direct-EMA recursion with fixed $\pi=.05$ and no LFU or HVP calculation.

The authoritative completed pilot is
`cache/mnist_experiment/rotated_mnist/phase3/rotated_mnist_phase3_low_data_pilot__replica-0001__ef901ae2610bfa01`.
It passed exact shared-initializer and pre-treatment metric checks. The initial
Fisher took 3.19 seconds to estimate, Lanczos realized all eight requested
directions without a numerical retry, and the complete paired run took 1,011
seconds. Most wall time came from repeatedly evaluating the 10,000-observation
fixed angle panel, not from learner updates.

The development trajectory supports the **promote** gate, with important
qualifications:

- normalized environmental-accuracy AUC was `.6153` for current-only and
  `.7216` for EWC, a paired lift of `.1063`;
- normalized environmental-NLL AUC was `6.580` for current-only and `1.388`
  for EWC, a paired reduction of `5.192`;
- EWC had greater current-angle accuracy at 34 of 41 evaluations, tied at the
  shared initial point, and was lower at 6 points;
- EWC reached persistent `.60` environmental accuracy after 280 observations,
  versus 320 for current-only;
- at $30^\circ$, upright-panel accuracy was `.753` for EWC and `.645` for
  current-only, while current-angle accuracy was `.615` and `.654`,
  respectively; and
- final current-angle worst-class recall was `.298` for EWC and `.210` for
  current-only, although individual-class recall effects were mixed.

The AUC gain is not merely upright retention. Over $0^\circ$--$15^\circ$,
current-angle accuracy AUC was `.792` for EWC versus `.614` for current-only;
over $15^\circ$--$30^\circ$, it was `.652` versus `.616`. EWC also reduced
expected calibration error from `.328` to `.100` on the first half and from
`.345` to `.221` on the second. The current-only learner's very large NLL
despite moderate accuracy is consistent with unstable, overconfident fitting
of eight-observation batches.

This remains one development replica. The negative endpoint contrast and
mixed classwise effects forbid a universal-dominance claim, while the broad
trajectory advantage forbids describing the result as retention only. The
pilot therefore motivates Phase 4 memory comparisons and later independent
confirmation. The optional $m$ sensitivity was not run: the two learners
neither saturated together nor failed to separate.

The computational tradeoff is visible. EWC used 2,212 optimizer function
evaluations and 9.87 seconds of learner optimization, versus 909 evaluations
and 3.42 seconds for current-only. Its online Fisher updates added only `.42`
seconds and the rank-8-plus-diagonal summary occupied 36,864 bytes. Thus the
observed statistical stability was purchased mainly with additional optimizer
work, not expensive Fisher maintenance.

The focused detachable suite now has 30 passing tests. The complete repository
suite has 348 passing tests and 2 skips, and the artifact-only results notebook
executes in approximately 3 seconds.

## Phase 4: Repeated-Path Memory and Replay Comparison

### Goal

Determine how bounded replay and EWC interact across return and revisitation,
using the complete six-knot path.

### Scope

1. Run current-only, fixed EWC, Replay B32, Hybrid B32, and fresh unbounded
   replay on the same full trajectories. Preserve replay arrival semantics and
   clean archive recursion from Plan 3.
2. Use the unbounded replay condition as a resource-unconstrained control. It
   stores every arrival once and samples minibatches without redundantly
   materializing transformed copies.
3. Compare first and second ascents at matched angles. Report return-leg
   recovery, revisit lift, upright retention, current-angle environmental
   multiclass accuracy and NLL, per-class precision and recall, worst-class
   recall, and every angle-specific fixed-panel metric. Retain digit-9 OvR
   fields only as supplemental continuity metrics.
4. Separate the applied contrasts into:
   - EWC versus current-only;
   - Replay B32 versus current-only;
   - Hybrid B32 versus Replay B32; and
   - bounded methods versus unbounded replay.
5. Report realized optimizer evaluations, learner/GPU time, replay bytes,
   Fisher bytes, and peak memory. Do not infer a resource advantage from
   nominal buffer size alone.

### Gate

Promote only contrasts that alter an applied decision: low-data transfer,
return/revisit memory, or the quality-cost frontier. A second-pass improvement
shared equally by every learner is exposure history, not evidence for EWC.

### Verification

- Buffers retain observations when they arrive regardless of whether a learner
  update is accepted or rejected.
- No method receives extra unique observations or evaluation feedback.
- Paired matched-angle contrasts are reconstructable from scalar artifacts.
- Plots contain no more than four conditions and include visible uncertainty
  once independent replicas exist.

### Check-in

Choose the practical memory conditions worth confirming and decide whether
the repeated path contains enough variation to justify an adaptive controller
challenge.

### Execution record

Phase 4 is complete. A strict Phase 4 schema and lockstep runner now execute
current-only, fixed-$\pi=.05$ EWC, Replay B32, Hybrid B32, and fresh unbounded
replay from one initializer and one exact transformed stream. Replay retrieves
each observation with its arrival-time angle; current observations enter only
after the learner update. Hybrid uses the accepted clean recursion in which
FIFO evictions alone advance a disjoint rank-8-plus-diagonal archive.

The predeclared six-knot smoke is
`cache/mnist_experiment/rotated_mnist/phase4/rotated_mnist_phase4_full_path_smoke__replica-0001__dc091fd0bdbe33ae`.
It passed every operational Go component: five complete conditions, exact
pairing, 40 verified arrival transforms, exercised B32 eviction and Hybrid
consolidation, disjoint replay/archive identities, no duplicate exposure,
finite diagnostics, realized rank eight, `.61%` peak GPU capacity, and an
exact match between the new one-pass metrics and the established evaluator on
512 fixed-panel observations. Its conservative full-run projection was 691
seconds, below the 60-minute limit.

The authoritative development artifact is
`cache/mnist_experiment/rotated_mnist/phase4/rotated_mnist_phase4_repeated_path_memory_screen__replica-0001__b85ea66a5617ad67`.
It completed 101 points and 100 paired updates in 225 seconds, verified all
800 arrival transforms, and repeated every smoke integrity result. Peak GPU
allocation was approximately 108 MB.

The principal whole-path results were:

| Condition | Accuracy AUC | NLL AUC | First-ascent accuracy AUC | Second-ascent accuracy AUC |
|---|---:|---:|---:|---:|
| Current only | `.5751` | `9.478` | `.4686` | `.6697` |
| Fixed EWC | `.7347` | `1.184` | `.7061` | `.7566` |
| Replay B32 | `.7407` | `6.560` | `.6837` | `.7870` |
| Hybrid B32 | `.7414` | `1.563` | `.7476` | `.7346` |
| Unbounded replay | `.8006` | `5.529` | `.7168` | `.8660` |

Fixed EWC and Replay B32 improved accuracy AUC over current-only by `.1596`
and `.1656`, respectively. Fixed EWC reduced NLL AUC by `8.294`, preserved
upright endpoint accuracy by an additional `.229`, and required only 44,456
optimizer-event evaluations. It is the strongest quality-cost point in this
development replica, not merely a retention mechanism.

Hybrid and Replay B32 had nearly identical aggregate accuracy AUC, differing
by only `.0007`. Hybrid reduced NLL AUC by `4.997`, improved first-ascent AUC
by `.0639`, and preserved an additional `.046` upright endpoint accuracy, but
lost `.0523` on second-ascent AUC and `.0185` at the final current angle. Its
mean matched-angle revisit accuracy change was slightly negative. The archive
therefore supplied early stability while impeding later reacquisition in this
trajectory; Hybrid is a consequential tradeoff, not a universal winner.

Fresh unbounded replay supplied the intended resource-unconstrained accuracy
ceiling. It improved whole-path accuracy AUC by `.0592` over Hybrid and final
current-angle accuracy by `.1534`. Its exact logical replay state was 2.55 MB,
approximately 21 times Hybrid's 32-event replay plus canonical archive state,
and it used about 1.98 million learner optimizer-event evaluations versus
approximately 255 thousand learner-plus-archive evaluations for Hybrid. Its
large NLL despite strong accuracy remains an overconfidence warning.

Second-pass improvement was not itself evidence for an EWC-specific memory
effect. Mean matched-angle accuracy lift was `.194` for current-only, `.048`
for fixed EWC, `.099` for Replay B32, `-.014` for Hybrid, and `.146` for
unbounded replay. Much of the positive revisit signal is therefore generic
additional exposure, while fixed EWC entered the revisit from a substantially
higher first-pass baseline.

The artifact-only results notebook now exposes four uncluttered views: EWC
isolation, bounded-replay isolation, the Hybrid/replay frontier, and matched
revisit lift, together with the realized resource ledger. Its 12 code cells
execute in 3.2 seconds and perform no training or repair.

**Gate decision:** retain all five paired conditions for later confirmation.
Fixed EWC, Replay B32, and the unbounded control clearly alter applied
decisions. Hybrid remains worth confirming because its calibration and
first-pass gains oppose its second-pass loss. The complete path produced
enough direction, speed, performance, and revisit variation to proceed to the
predeclared Phase 5 EDR reconstruction and challenge; this decision does not
claim that geometry or adaptive value has already been established.

## Phase 5: EDR Recommendation and Closed-Loop Challenge

### Goal

Test whether EDR offers prospective information about composition weight on a
path with direction changes and repeated exposure.

### Scope

1. Begin with an artifact-only reconstruction of $A_t$, $B_t$, instantaneous
   recommendations, EDR recommendations, and Plan 4 health diagnostics from
   the fixed-EWC Phase 4 trajectories.
2. Verify that the proposed EDR action is predictable, non-floor-driven, and
   meaningfully different across first ascent, return, and second ascent before
   training a closed-loop treatment.
3. Run paired EWC-only conditions for the predeclared fixed bracket
   $\pi\in\{.01,.025,.05,.10\}$ and EDR initialized at $.05$. Preserve
   $H_\pi=4$, $\pi_{\min}=.01$, and $H_\varphi=7.5^\circ$ as primary adaptive
   settings.
4. Treat fixed $.05$ as the prospective untuned baseline and the best tested
   fixed condition as a post facto research-only benchmark. Do not call the
   latter globally optimal, and do not claim EDR discovered a value supplied
   through its initialization, floor, or tuning grid.
5. Evaluate action timing against knots and predictive changes. An action that
   moves only after degradation is diagnostic, not corrective.
6. Evaluate adaptive value jointly through environmental-accuracy AUC,
   environmental NLL AUC, per-class precision and recall, worst-class recall,
   calibration, retention, and compute. Digit-9 OvR metrics remain
   supplemental. No one metric establishes optimal stochastic control.
7. Add paired fixed and EDR Hybrid B32 only if EWC-only EDR passes the
   predeclared predictive gate. Replay observations do not increase $m_t$ in
   the controller covariance model.

### Gate

- **Dynamic success:** EDR improves different path regions through coherent
  action changes and outperforms every tested fixed policy without a material
  secondary regression.
- **Automatic selection:** EDR approaches the post facto best tested fixed
  policy within the predeclared practical margin from cold-start $.05$, without
  task-specific trials or a material secondary regression.
- **Prospective value:** EDR improves over untuned fixed $.05$ and remains
  practically close to the best tested fixed policy, but does not demonstrate
  region-specific superiority.
- **Diagnostic only:** recommendations contain calibrated single-trajectory
  signal but closed-loop actions do not improve outcomes.
- **Stop:** actions are clipping-driven, delayed, unstable, or predictively
  harmful. Material underperformance relative to both fixed $.05$ and the best
  tested fixed policy is sufficient to reject practical use in this task.

### Verification

- Recomputed EDR decisions exactly match stored actions.
- Coefficients and decisions contain no current-batch leakage.
- The angular-distance gain is finite and lies in $[0,1]$ on every leg.
- No inverse, pseudoinverse, LFU, or HVP enters the treatment.
- Fixed and adaptive conditions share all non-controller state and seeds.

### Check-in

Classify EDR as dynamic, automatic-selection, prospective-value,
diagnostic-only, or stopped. Do not add replicas merely because an exploratory
trajectory is visually interesting.

### Execution record

Phase 5 is complete and stopped at its predeclared offline recommendation
gate. A detachable Phase 5 configuration, immutable artifact schema, and
reconstruction entry point now recover the predictable Fisher-risk
coefficients and EDR actions from the completed Phase 4 fixed-EWC trajectory.
The reconstruction uses only the prior accepted parameter/Fisher state when
forming each recommendation; it does not refit a model, inspect a current
outcome, invert a Fisher matrix, or actuate the counterfactual policy.

The authoritative reconstruction is
`cache/mnist_experiment/rotated_mnist/phase5/reconstruction/rotated_mnist_phase5_edr_challenge__replica-0001__bf5fa4f0d7f4f517`.
Its direct-EMA Fisher trace matched the Phase 4 source at every one of the 100
transitions, with maximum relative trace error `0.0`; all updates retained the
requested rank eight. A five-transition smoke reconstruction is retained at
`cache/mnist_experiment/rotated_mnist/phase5/reconstruction/rotated_mnist_phase5_edr_smoke__replica-0001__6ba65264c5d7176a`.

The full recommendations were finite, predictable, unclipped at the lower
bound, and not constant. After cold start, EDR ranged from `.03030` to `.05235`,
averaged `.04278`, and differed from fixed `.05` by mean absolute distance
`.00750`. The movement did not resolve into the path-specific response this
phase required, however. Mean actions were `.04260` on first ascent, `.04360`
on return, and `.04251` on second ascent, for a regional span of only `.00110`.
The direction reversal and repeated exposure therefore supplied no coherent
prospective actuation signal.

The gate classification is **stop**, not predictive failure: no Phase 5
closed-loop EWC or Hybrid learner was trained, so these artifacts make no
claim about counterfactual accuracy, NLL, calibration, or compute. The stop
avoided selecting a favorable interpretation after observing outcomes and
avoided spending the much larger closed-loop budget on a recommendation that
failed the prerequisite scientific test. Phase 6 has no promoted adaptive
contrast from Phase 5; reopening EDR requires a new, versioned environmental
challenge or controller hypothesis rather than extra replicas of this one.
The focused rotated-MNIST suite passes with 53 tests; the complete repository
suite passes with 371 tests and 2 skips.

#### Reopened offline sensitivity

The regional-mean gate is intentionally conservative, but it may suppress a
short-lived response around the two actual schedule changes. Before accepting
the stop decision, run one artifact-only sensitivity reconstruction on the
same fixed-EWC Phase 4 trajectory. This amendment was frozen before inspecting
the sensitivity results and does not authorize predictive metric selection.

Cross

$$
H_\pi\in\{1,2,4,8\}\text{ updates},
\qquad
H_\varphi\in\{1.875,3.75,7.5\}^\circ.
$$

The shorter $H_\varphi$ values jointly shorten trend memory and the existing
one-half-life cold start; this screen does not pretend to identify their
effects separately. For each setting, retain the complete recommendation
trajectory, cold-start duration, clipping rates, action span, total variation,
second-difference roughness, and the original regional means.

Replace broad regional means as the sole responsiveness diagnostic with a
causal knot comparison. At the dynamic knots $t\in\{40,60\}$, where angular
direction and speed change, compare the mean of the four decisions immediately
before the knot with decisions $t+1,\ldots,t+4$. Decisions at the knot itself
remain pre-change and are excluded from the post window. Use the same measure
at $t\in\{20,80\}$ as a same-direction, same-speed reference. Also record the
largest departure from the knot decision over the next eight predictable
actions and its lag.

Evidence of excessive action smoothing requires a shorter $H_\pi$ at fixed
$H_\varphi=7.5^\circ$ to increase mean dynamic-knot response by at least 50%,
without more than doubling second-difference roughness or reducing the ratio
of dynamic-knot response to same-speed-knot response. Report the complete grid
even when this criterion fails. Shorter cold start is informative only about
the initial leg and cannot by itself rescue the dynamic-controller claim.

This sensitivity may reopen a single predeclared closed-loop challenge only if
it identifies a finite, non-floor-driven recommendation with materially better
knot timing and a defensible smoothness tradeoff. Otherwise retain the Phase 5
stop result. No accuracy, NLL, calibration, or model outcome may enter this
offline selection.

The completed sensitivity artifact is
`cache/mnist_experiment/rotated_mnist/phase5/sensitivity/rotated_mnist_phase5_edr_challenge__replica-0001__bf5fa4f0d7f4f517`.
It reconstructed all 12 settings over the same 100-transition fixed-EWC path,
again matching every source Fisher trace exactly. No predictive outcome was
loaded or used.

Shortening action memory alone did not pass the frozen smoothing rule. At
$H_\varphi=7.5^\circ$, reducing $H_\pi$ from 4 to 1 increased mean
dynamic-knot response by `4.77x`, but increased second-difference roughness by
`3.59x` and still responded more at the same-speed knots than at the actual
direction/speed changes. Reducing $H_\pi$ to 2 nearly doubled both response and
roughness, while its mean dynamic response remained below the predeclared
`.003` materiality threshold.

The coupled trend/cold-start sensitivity was more informative. The clearest
descriptive setting used a fast trend and slow action,
$H_\varphi=1.875^\circ$ and $H_\pi=8$. It produced mean dynamic-knot response
`.01017`, same-speed-knot response `.00188`, and a response ratio of `5.41`,
with second-difference roughness `.00241`, no clipping, and three cold-start
steps. By comparison, the original $(H_\pi,H_\varphi)=(4,7.5^\circ)$ setting
produced `.00100`, `.00274`, `.36`, and `.00069`, respectively.

This result weakens the claim that EDR itself is unresponsive. It instead
suggests that the original trend memory was too slow and that complementary
timescales may be useful: a responsive trend estimate followed by a smoother
risk-action estimate. It does not yet identify whether shorter trend memory or
shorter burn-in caused the improvement, because the existing controller ties
both to $H_\varphi$. Under the frozen rule, the automated recommendation
therefore remains `retain_stop` for the present closed-loop challenge. EDR is
not rejected generally; a future retry should first separate trend memory from
cold-start duration, then predeclare one fast-trend/slow-action treatment.

#### Double-lap closed-loop retry

The next development experiment gives the fast-trend/slow-action controller a
single prospective closed-loop opportunity. It does not alter or overwrite the
completed reconstruction and sensitivity artifacts.

Use the three-leg double-lap path

$$
0^\circ\longrightarrow30^\circ\longrightarrow0^\circ
\longrightarrow30^\circ
$$

with 40 transitions per leg, 120 updates, and 121 pre-update evaluation
points. Set $m=4$, for 480 online observations: 40% fewer than the completed
Phase 4 path despite the finer temporal resolution. Compare two schedules
with identical endpoints, leg lengths, total angular distance, canonical
observation identities, and treatment seeds:

1. constant angular speed within every leg; and
2. a normalized logistic position curve with $\kappa=8$ within every leg.

The sigmoid schedule must attain each endpoint exactly. Its materially fast
central interval should last at least two $H_\pi$ half-lives, so this is a
responsive challenge rather than an action-timescale mismatch.

For each schedule, compare paired `current_only`, fixed-EWC
$\pi\in\{.025,.05,.075,.10\}$, and one EDR condition. EDR is frozen at
$H_\varphi=1.875^\circ$, $H_\pi=8$ updates, $\pi_{\min}=.01$,
$\pi_{\max}=.95$, and cold-start action $.05$. Decouple readiness from
spatial trend memory: release cold start after eight accepted updates, exactly
one $H_\pi$ half-life, while continuing to accumulate risk moments during
those updates. One realized $\pi_t$ controls both the EWC objective and Fisher
EMA. No LFU, inverse, pseudoinverse, replay, Hybrid, oracle path, or current
outcome enters the controller.

Primary controller evidence is the schedule-by-policy interaction

$$
\left(\operatorname{AUC}_{\mathrm{env}}^{\mathrm{EDR}}
-\operatorname{AUC}_{\mathrm{env}}^{\mathrm{fixed}}
\right)_{\mathrm{sigmoid}}
>
\left(\operatorname{AUC}_{\mathrm{env}}^{\mathrm{EDR}}
-\operatorname{AUC}_{\mathrm{env}}^{\mathrm{fixed}}
\right)_{\mathrm{linear}},
$$

reported against prospective fixed $.05$ and the post-facto best tested fixed
policy. Also report environmental NLL AUC, calibration AUC, fixed-panel
retention, worst-class recall, per-leg AUCs, optimizer work, Fisher work, and
complete action trajectories. Action diagnostics use lagged realized angular
speed because the current transition is unavailable when $\pi_t$ is chosen.

Classify the retry as:

- **dynamic value** when EDR coherently raises attention in fast regions,
  improves the sigmoid predictive outcome relative to fixed $.05$, and has a
  more favorable EDR-minus-fixed contrast on sigmoid than linear without a
  material secondary regression;
- **automatic-selection value** when EDR remains within `.01` environmental
  accuracy AUC of the post-facto best fixed policy on both schedules without
  dynamic superiority;
- **diagnostic only** when actions track speed/risk but predictive outcomes do
  not improve; or
- **stop** when actions remain unresponsive, unstable, or materially harmful.

This remains one development replica. It may promote a configuration but
cannot establish statistical significance.

The double-lap development run completed under the frozen design at
`cache/mnist_experiment/rotated_mnist/phase5/double_lap/rotated_mnist_phase5_double_lap_development__replica-0001__e8d49d4599638493`.
It contains 120 updates per schedule, 480 online observations, all six paired
conditions, exact stream-identity pairing, eight cold-start updates, finite
metrics, and realized rank eight throughout. The CUDA run required `656.08`
seconds. Its five-transition smoke artifact is retained separately at
`cache/mnist_experiment/rotated_mnist/phase5/double_lap/rotated_mnist_phase5_double_lap_smoke__replica-0001__12a32b2bbf9513de`.

The faster sigmoid regions produced a real but weak action signal. Mean EDR
action was `.2613` in above-average-speed regions and `.2393` elsewhere, a
lift of `.0220`, with lagged speed/action correlation `.0792`. The
EDR-minus-fixed-`.05` environmental-accuracy AUC contrast was also `.0782`
more favorable on sigmoid than linear. These observations preserve EDR as a
single-trajectory diagnostic hypothesis.

The realized controller was nevertheless predictively harmful. Post-cold
actions ranged from `.1277` to `.5325` on sigmoid and `.1582` to `.7309` on
linear, far above the tested fixed bracket. EDR environmental-accuracy AUC was
`.3394` on sigmoid and `.2728` on linear, compared with `.6198` and `.6314`
for prospective fixed `.05`; environmental NLL AUC increased by `34.35` and
`64.49`, respectively. Fixed `.025` was the post-facto best tested policy on
both schedules, with environmental-accuracy AUC `.6655` and `.7078`.

The coefficient audit explains the failure without suggesting a runner-scale
error. At $m=4$, noisy fitted displacements during and immediately after cold
start generate large Fisher-weighted trend-risk estimates. EDR interprets
that variation as environmental drift, raises $\pi_t$, weakens EWC, and then
observes still larger displacements. The frozen surrogate is being applied as
specified, but its local trend estimate is not calibrated well enough to
separate drift from finite-sample and optimizer variation in this regime.

The retry classification is therefore **stop with diagnostic signal**, not a
general rejection of adaptive composition. No EDR configuration is promoted
to Phase 6 from this development run. The negative closed-loop result and the
positive but weak speed response both remain part of the Plan 7 record.
The immutable run summary preserves its original automated label
`diagnostic_only`; a post-run gate audit found that the classifier had allowed
action responsiveness to outrank the predeclared material-harm stop rule. The
classifier and its regression test now enforce that stop-rule precedence;
the completed artifact itself was not modified.

#### Slow-trend single-lap retry

The double-lap failure motivates one final, narrower test of EDR as a slow
online composition tuner. This retry is versioned separately and does not
modify either completed Phase 5 artifact. Use

$$
0^\circ\longrightarrow30^\circ\longrightarrow0^\circ
$$

with 40 transitions per leg, 80 updates, $m=4$, and both constant-speed and
normalized-logistic ($\kappa=8$) schedules. Canonical observation identities,
labels, initialization, and non-treatment seeds remain paired across schedule
and policy. Compare only `current_only`, fixed EWC $\pi=.025$, prospective
fixed EWC $\pi=.05$, and `edr_slowtrend_slowaction`.

Freeze EDR at $H_\varphi=7.5^\circ$, $H_\pi=8$ updates,
$\pi_{\min}=.01$, $\pi_{\max}=.95$, and eight cold-start updates at $.05$.
One realized $\pi_t$ controls both the EWC objective and Fisher EMA. No LFU,
replay, Hybrid, inverse, pseudoinverse, oracle, or current outcome enters the
controller.

Changing $H_\pi$ alone is not expected to repair the double-lap release. On
the paired cold-start rows, increasing $H_\pi$ from 8 to 64 changed the first
post-cold recommendation only from `.1582` to `.1577` on linear and from
`.1397` to `.1403` on sigmoid, because the common EMA gain largely cancels in
$\bar A_t/(\bar A_t+\bar B_t)$. This retry instead tests slower estimation of
the trend entering the old-risk coefficient.

Classify **automatic-selection value** when EDR remains within `.01`
environmental-accuracy AUC of the better tested fixed policy on both schedules
without a material secondary regression. Classify **prospective value** when
it remains practically tied with fixed `.05` on both schedules and improves
at least one. Classify **diagnostic only** only when actions remain within
`[.01,.10]`, vary by at least `.005`, and avoid material predictive harm.
Classify **stop** when actions escape that practical bracket, become
unresponsive, or materially harm accuracy, NLL, upright retention, or
worst-class recall. This is one development replica and cannot establish
statistical significance.

The completed development artifact is
`cache/mnist_experiment/rotated_mnist/phase5/single_lap/rotated_mnist_phase5_slow_single_lap_development__replica-0001__30f2161a06eb8d50`.
It contains all 80 updates, 320 online observations per schedule, exact stream
pairing, eight cold-start updates, finite trajectories, predictable decisions,
and realized rank eight throughout. The CUDA run required `261.11` seconds.
The immutable smoke artifact is retained at
`cache/mnist_experiment/rotated_mnist/phase5/single_lap/rotated_mnist_phase5_slow_single_lap_smoke__replica-0001__974dff00a6acf408`;
its original automatic-selection label is a cold-only mechanical tie, and the
classifier now labels such smoke runs `integration_only`.

The retry classification is **stop**. Environmental-accuracy AUC was `.3603`
for EDR, `.5630` for fixed `.05`, and `.6331` for fixed `.025` on linear. On
sigmoid it was `.3973`, `.5200`, and `.5989`, respectively. EDR environmental
NLL AUC exceeded fixed `.05` by `33.38` on linear and `26.75` on sigmoid, with
material final upright-accuracy and worst-class-recall losses on both
schedules.

Slow trend estimation exposed a more specific timing failure. On sigmoid's
outward leg, EDR action averaged `.0512` and ranged from `.0294` to `.1502`;
its environmental-accuracy AUC was `.4986`, compared with `.4070` for fixed
`.05` and `.3855` for current-only. The action rose too late, however. After
the $30^\circ$ reversal its mean became `.4990` and its maximum `.6942`, while
return-leg accuracy AUC fell to `.2961` versus `.6330` for fixed `.05`. On the
constant-speed schedule, EDR was already elevated throughout both legs, with
means `.2881` outward and `.2673` on return.

Thus a longer $H_\varphi$ can suppress early noisy action on the gently
starting sigmoid leg, but it exchanges immediate variance for severe lag. The
discounted risk estimate accumulates a recommendation only near the first
endpoint, carries it across reversal, then enters the same weak-EWC/large-step
feedback loop. Neither the double lap nor this single lap promotes the current
EDR closed-loop policy to Phase 6. The outward-leg signal remains useful for
future theory about confidence-aware or change-point-aware controllers, but it
is not an applied result from Plan 5.

The artifact-only narrative and realized action plots are in
`mnist_experiment/rotated_mnist/single_lap_edr_results.ipynb`. It keeps the
linear and sigmoid schedules separate because this development phase has only
one replica and averaging the two interventions would erase the timing
contrast.

## Phase 6: Independent Confirmation of Promoted Contrasts

### Goal

Estimate uncertainty only for contrasts that survived the development gates.

### Scope

1. Freeze the promoted configurations, primary metrics, path windows, and
   minimum meaningful effects before adding replicas.
2. Start with five independent paired replicas, including the development
   replica only when it was generated under the final frozen schema.
3. Report paired mean differences and 95% Student-$t$ intervals over complete
   replica trajectories. Preserve per-replica paths and plot uncertainty around
   the mean at every angle.
4. Add replicas through `--resume` without changing earlier artifacts. Provide
   a precision report showing interval width for every primary contrast before
   recommending more compute.
5. Freshly execute the unbounded replay control in the confirmatory stream;
   do not splice a development artifact into a new pairing block.
6. Stop at the predeclared precision target or maximum replica count even if a
   preferred method remains inconclusive.

### Verification

- Configuration hashes are identical within each promoted condition across
  replicas except for replica identity and derived seeds.
- Pairing hashes match across treatments within a replica.
- Confidence regions use independent replicas, not steps or observations as
  independent units.
- The results notebook can identify missing replicas and emit a concise
  resumable command without running compute itself.

### Check-in

Accept, reject, or leave inconclusive each transfer, memory, and adaptive
claim. Decide whether additional replicas have enough decision value to
justify their cost.

## Phase 7: Findings Integration or Clean Retirement

### Goal

Turn the evidence into a concise scientific record while preserving the
option to discard an unproductive module.

### Scope

1. Produce `mnist_experiment/rotated_mnist/results.ipynb` as an artifact-only
   narrative with motivations, contracts, simple paired plots, uncertainty,
   cost metrics, negative results, and immutable artifact identifiers.
2. State separately whether Plan 5 established:
   - forward statistical efficiency;
   - retention only;
   - repeated-path memory;
   - an applied Hybrid advantage;
   - EDR automatic selection or prospective value;
   - dynamic EDR value; or
   - no useful treatment effect.
3. Integrate a finding into `mnist-findings.ipynb` only when its visual and
   statistical evidence supports the wording. Do not turn unchanged accuracy,
   Fisher reconstruction quality, or controller motion into an applied claim.
4. Amend [mathematical_overview.ipynb](../mathematical_overview.ipynb) only if
   the experiment exposes a missing mathematical condition or supports a new
   controller interpretation. Numerical accommodations remain in the
   appendix under the existing style.
5. If Plan 5 is unproductive, leave one concise historical decision in this
   plan and remove the detachable source, tests, small configs, and notebook
   in a separate user-approved change. Preserve immutable cache artifacts and
   never rewrite Plans 1--4.

### Verification

- Every headline conclusion links to an artifact-backed plot or paired table.
- Applied, diagnostic, and negative findings use distinct language.
- The notebook performs no training, data download, artifact repair, or hidden
  metric reconstruction.
- Removing the Plan 5 module leaves the legacy test suite and entry points
  unchanged.

### Final Check-in

Decide whether rotated MNIST becomes part of the primary experimental story,
remains a documented exploratory branch, motivates a separately planned task,
or is cleanly retired.
