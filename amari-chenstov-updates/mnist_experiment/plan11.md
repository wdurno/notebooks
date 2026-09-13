# Implementation Plan 11: Shrunk Local-Pi Recommendations

**Status:** Phases 0--2 completed; the original Phase 3 precision gate did
not open. The focused own-anchor follow-up below is specified, implemented,
and smoke-tested; no full confirmation replica has started. See
[the Phase 0--2 findings](rotated_mnist/plan11/PHASE0_2_FINDINGS.md).

Plan 11 asks a deliberately practical question: can a small, predictable
fraction of an online local-risk recommendation improve a low fixed
composition weight without inheriting the harmful action amplitude of the
unshrunk controller? It is an empirical policy test, not a new optimality
theorem or a repair of the Plan 10 pace inversion.

## Amendment: Focused Own-Anchor Follow-Up (2026-09-13)

The original Phase 2 decision and its stopped Phase 3 gate remain immutable.
This amendment specifies a **new, narrower** independent confirmation study;
it supersedes the original Phase 3 comparator, selection, precision cap,
schedule hierarchy, and Phase 4 decision rules **only for this follow-up**.
The historical plan below remains the record of how Phase 0--2 were run.
Nothing in the original development or benchmark replicas enters a
confirmation interval. The earlier sigmoid veto compared $(.025,.025)$ with
the *best* fixed control, not with its own $.025$ anchor; it is not erased by
changing the question. A same-anchor gain cannot establish superiority over
fixed $.01$ or an applied optimal-pi policy.

### Conditions And Estimands

The first and only authorized candidate for this follow-up is the live
closed-loop blend $(c,\lambda)=(.025,.025)$ versus fixed $\pi=.025$. Run
both policies on **linear** and **sigmoid** double-lap schedules, preserving
the model, data, action timing, cold start, and evaluation semantics above.
Treat the schedules as two distinct experimental conditions, not independent
replicas of one condition. For each schedule $s$ and fresh paired replica $r$,
the favorable difference is

$$
\Delta_{r,s}
=\operatorname{NLLAUC}_{r,s}(\pi=.025)
-\operatorname{NLLAUC}_{r,s}(c=.025,\lambda=.025),
\qquad s\in\{\mathrm{linear},\mathrm{sigmoid}\}.
$$

The two estimands are the schedule-specific **mean** paired whole-path
current-environment NLL-AUC gains, $\mathbb E[\Delta_{r,s}]$. Analyze each
schedule separately with a two-sided 95% Student-$t$ interval and unadjusted
two-sided $p$-value across independent replicas; a favorable result requires
positive mean gain and $p<.05$. Report both schedule results, interval widths,
replica win counts and medians, but do not treat a mean-effect test as proof
that most replicas benefit. These are two separate nominal 5% tests, **not**
a familywise-5% claim that at least one schedule works. No multiplicity-
adjusted threshold is imposed for this schedule-specific report.

The exploratory backup $(c,\lambda)=(.01,.025)$ versus fixed $\pi=.01$ is
**not** in the work ledger and must not start automatically. A failure or
inconclusive result for $.025$ triggers discussion, not backup execution.
Any backup confirmation requires a separately agreed, versioned protocol
and fresh data; it must not be folded into an unadjusted "any winner" claim.

### Compute And Resumption

Freeze **352 new replica identities for each schedule** before the first
follow-up trajectory. Use the same 352 replica indices and share each fresh
initialization, initial Fisher, and appropriate paired assets across the two
schedules; each schedule has its own arrival trajectory and two fully
recomputed policy trajectories. The statistical unit is a complete
fixed-versus-blend pair *within one schedule*. Neither steps nor the two
schedules multiply the independent replica count. Use seeds disjoint from
development and historical replicas, and write a new versioned configuration,
condition ledger, and artifact namespace without modifying completed runs or
the stopped development decision.

This initial target is $352\times2\times2=1{,}408$ full trajectories.
At the Phase 0 rates of 58.25 seconds per trajectory and about 10.82 seconds
of shared setup per replica, the projection is **23.8 hours** for both
schedules and policies, excluding unpredictable overhead. The 352-replica
target is an optimistic mean-effect planning calculation based on the
observed *linear* pilot gain; it does not promise 80% power for sigmoid's
smaller observed mean gain, nor for a $.02$ gain on either schedule. The
earlier roughly 101-hour figure was a separate, **uncommitted** sigmoid-only
projection for 2,848 replicas at its observed pilot mean effect, not part of
the 23.8-hour initial target.

Build `--resume` and bounded-session execution before starting long compute.
The immutable ledger is keyed by follow-up version, replica, schedule, and
policy. Work may be allocated in any convenient order or batch size between
linear and sigmoid; interrupted trajectories restart from their frozen inputs,
completed trajectories are never changed, and a completed pair contributes
only to its own schedule. Intermediate inspections may show per-schedule
completion counts, integrity, failures, resources, and clearly labeled
descriptive paired-gain plots. They must not declare significance or change
the frozen 352-replica target because one interim effect looks promising.
The entry point is `python -m mnist_experiment.rotated_mnist.plan11.followup`;
`--schedule linear` or `--schedule sigmoid` selects the next condition,
`--max-pairs` and `--max-wall-seconds` bound a session, and every invocation
after the first uses `--resume`. The focused progress notebook is
`rotated_mnist/shrunk_pi_results.ipynb`.

After all 352 planned pairs for a schedule are complete, its fixed-size
inference may be reported even if the other schedule is still running; keep
the latter visibly incomplete and report both eventually. Do not repeatedly
test a cumulative sample and stop when $p<.05$. If further compute seems
worthwhile after inspection, first agree a **new independent, fixed-size
follow-on cohort** and analyze its conventional $p$-value on that new cohort,
or separately predeclare a valid sequential design. Do not silently append
outcome-selected replicas to the initial 352 and call the resulting ordinary
Student-$t$ $p$-value confirmatory.

### Artifact-Only Notebook

Amend the Phase 4 notebook for this follow-up so it validates the frozen
ledger and completed artifact hashes, then renders both schedules' progress
and descriptive trajectories even while either ledger is incomplete. For
each schedule, average the existing per-exposure current-environment NLL and
accuracy trajectories over the **same completed fixed-versus-blend replica
pairs**, plotting both policies on their shared pre-update exposure clock.
Also show the corresponding running exposure-normalized NLL AUC and accuracy
AUC, computed with the same trapezoidal convention as the terminal summary;
validate that the final running values match the recorded AUCs. Keep linear
and sigmoid in separate panels, and label all partial-sample curves
descriptive. These plots read metrics already produced by every trajectory:
they require no additional evaluation or learner run. Unlock
a schedule's final interval and $p$-value only after **all 352** of its
predeclared pairs pass integrity and pairing checks. A missing or corrupt
artifact is an explicit incomplete/failure state, never a silently dropped
replica. Keep the original Phase 2 discovery plots and stopped decision
visually separate from the new confirmation results. The notebook only reads
artifacts; it never trains, resumes, downloads, or repairs a run.

## Evidence And Scope

- [Plan 2](plan2.md) and [Plan 3](plan3.md) support fixed `.05` EWC and fixed
  Hybrid B32 on their tested low-data path. They do not establish an online
  optimal-pi controller. Lowering the Plan 2 plug-in floor to `.01` destabilized
  that feedback loop; it did **not** test a fixed `.01` learner.
- In the one-replica Rotated-MNIST development studies, fixed `.025`
  descriptively beat fixed `.05`, while both legacy EDR and the [Plan 8](plan8.md)
  decomposed controller could overshoot and damage prediction. The latter
  correctly separates the tracked covariance state from the uncertain
  movement premium, but did not survive double-lap reversal stress.
- [Plan 7](plan7.md) attributed much of the recommendation error to an inflated
  movement numerator and a mismatch between population movement and realized
  learner catch-up. Shrinking an action can reduce amplitude; it cannot by
  itself correct a delayed or wrong-direction signal.
- [Plan 10 Phase 1d](plan10.md#amendment-phase-1d-direct-finite-ewc-response-surface)
  found mean-risk minimizers below `.05` at every tested in-bound anchor/pace
  cell. They ranged from `.0125` to `.0375`; `.01` itself was not tested.
  Those are one-step fits at frozen `.05` histories, not closed-loop evidence
  for any Plan 11 policy.

The narrow candidate is EWC-only on the existing `m=4` Rotated-MNIST
double-lap, with the `linear` schedule primary and the existing normalized
`sigmoid` schedule a reversal/timing stress. Keep the canonical 512-parameter
CNN, 30,000-observation upright initialization, 20,000-score initial Fisher,
rank-8-plus-diagonal direct EMA, 50 L-BFGS iterations, 120 transitions, fixed
evaluation panels, and Plan 5 transforms and exposure semantics. Do not
cross this study with replay, Hybrid B32, LFU, pace control, a new model, or
batch-size changes. No Plan 10 reference path or oracle may select actions.

## Scientific Contract

The input recommendation is the Plan 8 decomposed-EDR **predictable estimate
of the marginal one-step Fisher-risk minimizer**, recomputed on each policy's
own pre-transition state:

$$
\widetilde\pi_t
=\frac{\bar\rho_t+m q_t}{\bar\rho_t+m q_t+1},
\qquad m=4.
$$

Here $\bar\rho_t$, the trend, residual scale, Fisher summary, and $q_t$ retain
the Plan 8 definitions and timing. This is not the retrospective conditional
oracle, the population marginal oracle, or the empirical minimizer of
next-angle NLL. Historical names containing *optimal* do not change that
interpretation.

Explore the fixed anchors and shrinkage gains

$$
\mathcal C=\{.01,.025,.05\},
\qquad
\Lambda=\{0,.025,.05,.10,.20,1\}.
$$

For each $(c,\lambda)\in\mathcal C\times\Lambda$, define

$$
\pi_t^{(c,\lambda)}
=\operatorname{clip}_{[.01,.95]}
\left[c+\lambda(\widetilde\pi_t-c)\right],
\qquad 0\leq\lambda\leq1.
$$

The values $\lambda=.025,.05,.10,.20$ are exploratory development
treatments, not four confirmatory hypotheses. The endpoints have exact
interpretations: $\lambda=0$ is fixed $\pi_t=c$, while $\lambda=1$ applies
the unshrunk predictable recommendation. During the first eight accepted
updates, set $\pi_t=c$ for every member of the same-anchor family while the
estimator accumulates support. Thus comparisons within one $c$ differ only
in the post-cold shrinkage gain. The unshrunk $c=.05$ condition is compatible
with the historical Plan 8 cold start in principle, but its new trajectory
must still be computed on the new replica. For $c=.01$ or $.025$, it is a
new cold-start treatment, not a replay of a historical oracle.

The grid is deliberately small and logarithmically weighted toward strong
shrinkage. The historical Plan 8 double-lap linear mean recommendation was
approximately $.292$; for example, $(c,\lambda)=(.01,.05)$ would map that
single number to approximately $.024$. This motivates including the pair,
not privileging it or predicting its new closed-loop mean. The development
stage will nominate **one** $(c^\dagger,\lambda^\dagger)$ using a frozen
selection rule. Its independent confirmation, rather than the development
grid, determines whether any benefit is real. A later outcome-driven
revision of the grid, half-lives, clip, or cold-start rule requires a new
plan and fresh confirmation data.

At every transition, one realized action must weight the EWC objective, the
direct-EMA Fisher update, and the exact composition recursion

$$
q_{t+1}=(1-\pi_t)^2q_t+\pi_t^2/m.
$$

The action $\pi_t$ must be fixed before the new four-observation batch is
seen. The batch, accepted displacement, Fisher update, $q_t$ update, residual
moments, and all future recommendations are recomputed from each policy's own
trajectory. Offline blending of stored Plan 8 actions is a diagnostic only;
it is not a counterfactual policy evaluation. Record the raw recommendation,
blended action, clipping/fallback status, cold-start status, and every
intermediate coefficient needed to reconstruct a decision.

## Paired Design And Outcomes

Each independent replica receives a fresh initialization fit, initial Fisher,
partition, stream, and evaluation sample, with separately named component
seeds. Within a replica, all conditions share the initialization, initial
Fisher, schedule, arrival identities/order, evaluation panels, and
non-treatment optimizer settings. Both schedules are evaluated within the
same replica; they are not counted as two independent statistical units.
Development may use completed historical runs for mechanical compatibility
checks, never as independent confirmation. No new treatment resumes a
historical post-initialization model, Fisher, controller, or optimizer state.

The primary endpoint is exposure-normalized, whole-path current-environment
multiclass NLL AUC on the **linear double-lap**. Development nominates one
blend $(c^\dagger,\lambda^\dagger)$ and one fixed comparator $b^\dagger$:
the fixed $c\in\mathcal C$ with the lowest development mean linear NLL AUC.
Both identities are frozen before fresh confirmation. For replica $r$, define
the favorable paired difference

$$
\Delta_r^{(b)}
=\operatorname{NLLAUC}_r(b)
-\operatorname{NLLAUC}_r(c^\dagger,\lambda^\dagger),
$$

where $b$ is a named control. The primary contrast uses $b=b^\dagger$;
the other fixed actions and the unshrunk policy with the **same**
$c^\dagger$ test whether attenuation adds value beyond a low constant and
the original recommendation. Fixed $.025$ is the strongest previously
tested low fixed weight on this rotated development path, but that is
one-replica evidence and does not justify treating it as a universal winner.
Report every available paired difference even if the primary contrast fails.

Secondary outcomes are whole-path and per-leg environmental accuracy and NLL
AUC, digit-wise/worst-class recall, final upright-panel retention,
calibration, action variation and floor occupancy, optimizer acceptance,
Fisher diagnostics, and learner time. On sigmoid, report the same paired
outcomes and the action's response and release around both reversals. A mean
gain that conceals a catastrophic leg or a delayed high-action tail is not an
applied success. Every metric is evaluated on the same pre-update exposure
clock across conditions.

## Phases And Gates

### Interruption And Resumption Contract

Implement `--resume` before the Phase 0 benchmark and use it for the Phase 1
smokes, Phase 2 development, and Phase 3 confirmation as well. Each invocation
must use a durable, frozen work ledger keyed by phase, replica, schedule, and
policy configuration. Persist the shared initialization, initial Fisher,
stream, and evaluation-panel identities once per replica; publish each
completed trajectory atomically with a validated `COMPLETED` marker. On
`--resume`, verify the configuration and source hashes, schema, seeds, paired
identities, phase decision artifacts, and planned work ledger; skip completed
units and run only missing units. A halted or corrupt in-progress trajectory
must restart from its beginning using the same frozen inputs and seeds, never
contribute partial metrics, and never alter a completed run. This bounds lost
learner work to one trajectory rather than one replica or phase. Refuse to
resume if the scientific contract changed; require a new versioned run
instead. Offer an optional `--max-wall-seconds` limit that stops at a
trajectory boundary, making short compute sessions practical.

### Phase 0: Frozen Contract And Cost Audit

1. Version the Plan 11 configuration, action, metric, and artifact schemas in
   a detachable module under `rotated_mnist/`. Preserve all Plan 5--10
   sources and completed artifacts. Record the exact source hashes used for
   mechanical development checks.
2. Before any long run, freeze $\mathcal C\times\Lambda$, the two schedules,
   the exact shortlist and nomination rules below, primary metric, safety
   margins, independent replica seeds, maximum replica count, and cost
   ceiling. Configurations must serialize both $c$ and $\lambda$ explicitly;
   names alone must not encode ambiguous decimal places.
3. Benchmark one full paired replica's runtime and storage. Project the
   development grid (at most 300 full trajectories) and largest permitted
   confirmation (2,560 full trajectories) separately before requesting
   execution. Historical Plan 5/8 double-lap runs suggest roughly 55--60
   seconds per trajectory, or around five and 40 hours of learner work at
   those respective maxima, **before** fresh initialization/Fisher cost.
   Provisionally reserve six to eight hours of serial GPU wall time for
   Phases 0--2 together, including the benchmark, smoke checks, fresh
   replica setup, and run overhead; replace this estimate with the Phase 0
   measurement before the full development grid. These are planning
   estimates, not an execution authorization. No confirmation begins merely
   because a smoke run completed.

**Gate:** stop if actions cannot be reconstructed from pre-transition state
or the projected study exceeds the reviewed compute ceiling. A numerically
invalid fixed-anchor smoke excludes that $c$ and all of its blends, with the
exclusion recorded before development; continue only if at least two anchors
remain. Never replace an excluded value. Changing $\mathcal C$, $\Lambda$,
or the cold-start convention after seeing outcomes is a new experiment.

### Phase 1: Implementation And Mechanical Smoke

Use a tiny immutable smoke at each $c\in\mathcal C$ to check $\lambda=0$
exactly reproduces fixed $c$, $\lambda=1$ exactly reproduces the Plan 11
unshrunk rule with cold start $c$, and every interior blend lies between
them before clipping. Verify shared arrival identities,
independent policy state, no current-batch leakage, exact $q_t$ recursion,
cross-consumer action parity, failed-fit accounting, completion markers, and
resumption after both a clean stop and an interrupted trajectory without
mutation of a completed run. Compare resumed artifacts and metrics with an
uninterrupted run under the same configuration, excluding timestamps and
runtime counters. A smoke is not evidence of predictive value.

### Phase 2: Development And Precision Feasibility

Run 12 **new** paired development replicas on the linear schedule for the
retained part of the $\mathcal C\times\Lambda$ grid. With all anchors valid,
this is 18 policy conditions, including three fixed baselines and three
same-anchor unshrunk controls.
Every condition starts from the same replica initialization and sees the
same arrivals, but recomputes its own full feedback trajectory. Report every
grid result and failure; the smallest observed NLL is a development
selection statistic, not a treatment effect estimate.

The shortlist rule is fixed before execution:

1. Every retained fixed control must complete all 12 replicas; otherwise stop.
   An interior pair $(c,\lambda)$ is eligible only if it
   completes all 12, has at most 10% unsupported-scale fallback after cold
   start, and does not exceed its own fixed-$c$ mean NLL AUC by more than
   $.10$ or trail its own fixed-$c$ mean accuracy AUC by more than $.02$.
   Failures and rejected pairs remain in the ledger; no replica is replaced.
2. Within each $c$, identify the minimum eligible mean linear NLL AUC and
   nominate the smallest $\lambda$ within $.01$ of that minimum. Retain up
   to two anchor winners with lowest mean linear NLL AUC, breaking exact
   ties by smaller $c$. This preserves an anchor comparison while limiting
   sigmoid compute. If no winner exists, stop.
3. On the **same 12 development replicas**, run the sigmoid schedule for
   the retained one or two blends, every retained fixed control, and the
   same-anchor unshrunk controls. Every retained fixed sigmoid control must
   complete. The sigmoid results may veto a finalist but cannot reorder
   them by favorable sigmoid gain. Veto a finalist if its mean sigmoid NLL
   AUC exceeds the lowest fixed sigmoid mean by more than $.10$, its mean
   sigmoid accuracy AUC trails the highest fixed sigmoid mean by more than
   $.02$, or it has a nonfinite/failed trajectory.
4. Define $b^\dagger$ as the fixed anchor with lowest mean **linear** NLL
   AUC among the retained development controls. Among non-vetoed finalists,
   choose the lower linear-NLL finalist as $(c^\dagger,\lambda^\dagger)$.
   Stop before confirmation if it trails $b^\dagger$ by more than $.02$ mean
   linear NLL AUC. Exact ties use smaller $\lambda$, then smaller $c$.

These 12 replicas are discovery units only. A favorable development grid
does not establish significance, and no development trajectory enters the
final interval. Report action timing, reversal release, per-leg loss,
calibration, and whether the selected rule merely behaves as fixed $c$.

For a small-effect trial, use **variance, not the development effect sign**,
to size fresh confirmation. Let $s_{\rm upper}$ be the largest one-sided
90% upper standard-deviation bound for the non-vetoed finalists' paired linear
NLL-AUC differences against $b^\dagger$ on the 12 development replicas.
Predeclare $\delta_{\rm design}=.02$ NLL AUC (about 0.7% of the historical
fixed-$.05$ linear NLL AUC, solely a power target), two-sided type-I error
$.05$, and power $.80$. Set

$$
n_{\rm required}
=\left\lceil
\frac{(z_{.975}+z_{.80})^2s_{\rm upper}^2}
{\delta_{\rm design}^2}
\right\rceil,
$$

then round up to a whole block of 16, with at least 32 and at most 256 fresh
replicas. This normal approximation is a planning calculation, not the final
analysis. It powers only the primary comparison against $b^\dagger$, not
the conjunction of all fixed-policy and safety comparisons. If the required
count exceeds 256 or its timed cost exceeds the reviewed ceiling, stop as
**not resolvable at this budget**. Do not lower the effect target, reuse
development replicas, or peek for a favorable sign to rescue the design.
Freeze the selected pair, comparator, resulting $n$, and full seed/condition
ledger before the first confirmation run.
Write a versioned development decision artifact recording the complete grid,
shortlist and veto outcomes, cost/precision estimate, and whether the Phase 3
gate opened. An early stop remains a reportable development result, not a
missing confirmation run to fill in later. Evaluate shortlist and veto gates
only after every planned unit of the relevant stage is complete, then freeze
the decision artifact before scheduling the next stage; resumption must reuse
that decision rather than recompute it from a partial grid.

### Phase 3: Fixed-Size Independent Confirmation

Run exactly the frozen $n$ new replicas on both schedules for the nominated
$(c^\dagger,\lambda^\dagger)$, the retained fixed anchors, and the unshrunk
rule with cold-start anchor $c^\dagger$. New replicas must have new initial
fits and Fishers; no
development or historical trajectory contributes to an inferential interval.
Do not stop early for significance or extend after seeing the final effect.
Check blocks only for completeness, pairing, safety failures, and resource
limits. If a resource limit interrupts execution, label the experiment
incomplete rather than report a selected positive subset. `--resume` must
continue the original frozen replica/condition ledger until it is complete;
it must not replace failed replicas, change $n$, or treat the first completed
subset as the confirmation sample.

Estimate paired mean differences and two-sided 95% Student-`t` intervals over
independent replicas; show per-replica differences and a replica-bootstrap
sensitivity interval. Do not treat steps, batches, schedules, or four actions
within one replica as independent samples. Report unadjusted primary and
secondary effect sizes, interval widths, signs across replicas, all failures,
and per-schedule/leg distributions. Secondary plots cannot revise the primary
endpoint or choose a new gain.

### Phase 4: Decision

Create `rotated_mnist/shrunk_pi_results.ipynb` as the artifact-only summary
of this study. It must load and validate the completed Phase 2 runs and
development decision artifact; when Phase 3 was authorized, it must also
require the frozen confirmation ledger and every planned completed run.
An explicit Phase 2 stop should render a development-only report prominently
labeled **not independently confirmed**; missing or incomplete authorized
confirmation runs, incompatible schemas, or pairing mismatches must instead
raise a clear error. The notebook must never train, download data, or repair
artifacts.

Show the full development $(c,\lambda)$ grid with failed/excluded conditions,
the selected blend and fixed comparator, action and raw-recommendation
trajectories, and the precision/cost gate. If confirmation completes, show
paired replica-level primary differences and intervals, comparisons with all
fixed and same-anchor unshrunk controls, whole-path and per-leg outcomes on
both schedules, reversal timing, safety/failure diagnostics, and learner time.
Keep exploratory grid plots visually separate from independent confirmation
and state the decision below directly from the frozen rules.

Classify the nominated blend as follows:

- **Small adaptive benefit:** the linear primary paired 95% interval against
  $b^\dagger$ lies above zero and its point NLL-AUC gain is at least $.02$.
  Its paired interval against the same-anchor unshrunk control must also lie
  above zero; if that control fails numerically, report harm reduction as a
  failure contrast rather than a confidence interval. For an applied
  adaptive claim, simultaneous 95% paired lower bounds
  against **all retained** fixed controls must exceed zero, resampling complete
  replicas together. Sigmoid must be noninferior to $b^\dagger$: the one-sided
  95% upper bound on its NLL-AUC regression is below `.10`, and the one-sided
  95% upper bound on its environmental-accuracy AUC regression is below `.01`.
  The one-sided 95% lower bounds on final upright accuracy and worst-class
  recall differences must exceed `-.02` against $b^\dagger$ on each
  schedule. Mean learner-only time, including recommendation computation,
  must not exceed $b^\dagger$ by more than 10%; no blend failure is
  eligible for promotion. The broader interval requirements may leave a
  primary positive result unresolved for application at the sample cap.
- **Detectable but not useful:** the primary interval excludes zero but the
  effect misses `.02`, a cheaper fixed policy matches it, or safety/resource
  margins fail. Report the effect without promoting adaptive control.
- **Unresolved:** intervals include zero or the planned precision/cost gate
  cannot be met. Do not turn this into a positive or negative theorem.
- **Rejected for these paths:** the blend is materially worse, remains
  mistimed around reversals, or reproduces the unshrunk feedback pathology.

An apparent win over $\lambda=1$ alone is harm reduction, not adaptive value.
If any fixed action wins, that is a fixed-composition finding, not
validation of the local recommendation. If only the sigmoid schedule benefits
while linear fails, retain the schedule interaction as a new hypothesis; do
not relabel the frozen primary test.

## Artifacts, Analysis, And Boundaries

Computational work belongs in a Python entry point with immutable
configuration-driven runs under a new
`cache/mnist_experiment/rotated_mnist/plan11/` namespace. Record configuration
hashes, independent component
seeds, git/runtime metadata, source identities, all scalar trajectories,
initialization and Fisher diagnostics, action ingredients, optimizer failures,
resource use, and run status. Publish `COMPLETED` only after required files
are flushed and validated. Add a tiny CPU smoke and focused unit tests for
blending/clipping, temporal predictability, cold start, `q` recursion, paired
configuration generation, immutable collisions, interrupted-run restart,
and resumed-versus-uninterrupted equivalence. The results notebook loads
completed artifacts only and never trains or repairs a run.

Keep the historical Plan 8 cold-`.05` policy, retrospective population
oracles, and Plan 10 Phase 1d surface as context, not confirmatory controls.
Even a confirmed small gain would be conditional on this Rotated-MNIST model,
batch size, schedules, and computational budget. It would justify studying
whether shrinkage transports to the accepted Hybrid B32 application; it
would not itself establish a globally optimal or generally deployable pi.
