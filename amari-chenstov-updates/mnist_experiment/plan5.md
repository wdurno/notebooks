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
| 2 | Learnability and geometric feasibility audit | Pending |
| 3 | Low-data transfer pilot | Pending |
| 4 | Repeated-path memory and replay comparison | Pending |
| 5 | EDR recommendation and closed-loop challenge | Pending |
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

- current-angle digit-9 OvR accuracy, precision, and recall as the primary
  transfer-sensitive outcomes;
- current-angle environmental multiclass accuracy and NLL as broader task and
  likelihood context;
- digit-9 OvR accuracy, precision, recall, environmental multiclass accuracy,
  and NLL reported separately at each angle in the fixed
  $\{0^\circ,15^\circ,30^\circ\}$ panel, never pooled into one unspecified
  `accuracy`;
- upright digit-9 OvR accuracy, precision, recall, environmental multiclass
  accuracy, and NLL as retention outcomes;
- secondary per-class precision, recall, and OvR accuracy for digits 0 through
  8 so a digit-9 gain cannot hide broad class failure;
- ECE and Brier score;
- cumulative current observations and optimizer evaluations;
- learner time, GPU time, persistent bytes, and peak memory; and
- replay occupancy, Fisher-summary bytes, and total stored observation bytes.

For digit 9, define one-versus-rest accuracy explicitly as

$$
A_{9,\mathrm{OvR}}(n)
=
\frac{TP_9(n)+TN_9(n)}{P_9(n)+N_9(n)}.
$$

Precision and recall remain beside it because OvR accuracy can be dominated by
the non-nine majority. Define the primary first-ascent transfer summary against
cumulative current observations as

$$
\operatorname{AUC}_{9,\uparrow1}
=
\frac{1}{n_{\uparrow1}}
\int_0^{n_{\uparrow1}}
A_{9,\mathrm{OvR}}^{\mathrm{current}}(n)\,dn,
$$

calculated by the trapezoidal rule over stored evaluation points. Define
$\operatorname{AUC}_{\mathrm{env},\uparrow1}$ analogously for environmental
multiclass accuracy, but do not let environmental AUC substitute for the
digit-9 transfer estimand.

For the adaptive comparison, let

$$
Y_0
=\operatorname{AUC}_{9,\uparrow1}^{\pi=.05},
\qquad
Y_\star
=\max_{\pi\in\{.01,.025,.05,.10\}}
\operatorname{AUC}_{9,\uparrow1}^{\pi},
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
$Y_\star-Y_{\mathrm{EDR}}\leq.01$, together with no material precision,
recall, environmental-accuracy, retention, calibration, or resource
regression. Report those guardrails separately rather than combining them into
one score.

Define first-pass transfer efficiency through complete trajectories, not a
single noisy crossing. Report each mean paired trajectory, its 95% Student-$t$
interval, the two named AUCs, and an observation-efficiency curve relative to
current-only. Threshold-crossing summaries are secondary and require
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

- **Transfer success:** EWC materially improves
  $\operatorname{AUC}_{9,\uparrow1}$ or persistent digit-9 observation
  efficiency over current-only, with coherent precision and recall, while
  preserving upright and environmental capability. Environmental accuracy
  may strengthen this finding but cannot establish transfer success alone.
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
- Froze digit-9 OvR accuracy AUC over cumulative first-ascent observations as
  the primary transfer metric, accompanied by precision and recall.
  Environmental accuracy remains required context and cannot substitute for
  the transfer estimand.
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
   record attainable current-angle digit-9 OvR accuracy, precision, recall,
   environmental multiclass accuracy, NLL, calibration, displacement, and
   Fisher summaries.
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
4. Measure $\operatorname{AUC}_{9,\uparrow1}$ as the primary transfer summary,
   with digit-9 precision and recall beside it. Report
   $\operatorname{AUC}_{\mathrm{env},\uparrow1}$, upright retention, the
   angle-specific panel metrics, NLL, calibration, expected observations to
   persistent thresholds, optimizer work, time, and memory as context and
   guardrails.
5. Inspect complete trajectories before reducing them to threshold or endpoint
   tables. Accuracy barely changing is a negative result, not preservation of
   applied value.

### Gate

- **Promote:** EWC shows a material paired low-data advantage over current-only
  in digit-9 OvR AUC or persistent observation efficiency, with coherent
  precision and recall, while preserving upright and environmental
  performance.
- **Retention only:** EWC preserves upright capability but does not improve
  digit-9 learning on the rotated environment. Record that result and do not
  claim statistical efficiency even if environmental accuracy improves.
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
   recovery, revisit lift, upright retention, current-angle digit-9 OvR
   accuracy/precision/recall, environmental multiclass accuracy, and every
   angle-specific fixed-panel metric.
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
6. Evaluate adaptive value jointly through digit-9 OvR AUC, its precision and
   recall, environmental accuracy, NLL, calibration, retention, and compute.
   No one metric establishes optimal stochastic control.
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
