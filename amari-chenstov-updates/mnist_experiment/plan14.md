# Implementation Plan 14: Akaike Score Gating For Adaptive Layer Unfreezing

> **Hypothesis:** a continual learner can adapt the number of active parameters
> to the information and representational demands of incoming data, achieving
> useful predictive performance with few new observations and limited compute.
> Opening layers when their estimated likelihood improvement exceeds their
> additional fitting optimism may balance the persistent bias of excessive
> freezing against the estimation cost of insufficient freezing. Accumulated
> evidence and accurate cross-layer Fisher geometry may make this signal useful
> even when individual arrival batches are very small.

**Status: Complete, 17:10 PDT October 16.** All **2,304 distinct
final trajectories** are sealed (32 paired replicas in each of 72
policy/condition cells). Phase 6 closeout and driver provenance are sealed;
the main worker and disk supervisor exited 0. Independent checks verified
the complete cohort, required closeout/invocation checksums, and unchanged
31 frozen scientific hashes. There are 5,736 sealed scientific units and
47.16 GiB free after final tests. The report-only Phase 3 supplement completed
its full integrity scan and exited 0; all four inserted plots were visually
reviewed. The final notebook has 29 cells, 164 figure outputs, correctly
ordered phases, zero error outputs, and an artifact-grounded Phase 6
Conclusion whose primary effect table was checked against the sealed data.
All **411 unit tests passed again in 85.77 seconds**. The writer, supervisor,
reporter and test processes have exited, the run lock is free, and the GPU
has no compute process. No planned compute remains; do not restart science or
add replicas without a new specification. Final provenance and verification
details are in [`akaike_score_gate/HANDOFF.md`](akaike_score_gate/HANDOFF.md).

The selected-policy primary mean NLL-AUC comparison favors fixed controls
in three conditions; the m18 mixture comparison is inconclusive. Secondary
factorial results support more accurate current-gate curvature, especially
for rotations, but better-learning rotation variants exceed the mean-retention
bound. These are tradeoffs, not a primary adaptive-policy victory; see the
closeout's paired intervals and the handoff's concise result table. Preserve
the distinction between selected primary comparisons and descriptive or
factorial secondary findings, and retain all unresolved optimizer states.

**Historical continuation, 07:00 PDT October 16.** Actual host preflight found the
lock free and prior PID 6366 absent after the preceding turn was interrupted.
CUDA float64 works in `~/.venv`; all 31 frozen scientific hashes and the
tested reporting-driver hash match. The 411-test passing result from the
preceding continuation remains applicable: only Markdown changed since it
ran. The new actual held-lock writer is PID 4867 (native session 34866),
with independent disk supervisor session 5168 requesting SIGTERM below
22 GiB free. Free disk at launch was 52.70 GiB; preserve the 20-GiB reserve.
Phase 5 resumes from **1,875 of 2,304 distinct final trajectories** sealed
(429 remain), with replicas 0--25 complete across all four conditions.
The partial replica-26 linear `depth4_diag` checkpoint is step 106, stage
`fit`, depth 4; accepted operations, archive reservation and budget are
preserved. Continue the unchanged cohort through Phase 6, then apply the
reviewed report-only Phase 3 supplement after the final ordinary refresh.
Restart validation/replay still costs time: the preceding tested-driver
continuation reached its first recovered seal in 62.46 minutes. Notebook
refreshes, controls, scientific settings and stopping rules remain unchanged.
Current identities, provenance, and checkpoint details are in
[`akaike_score_gate/HANDOFF.md`](akaike_score_gate/HANDOFF.md).

**Progress, 09:34 PDT October 16:** Phase 5 has **1,950 of 2,304** distinct
trajectories sealed (354 left), with 27 paired replicas complete across each
of the four conditions. The recovered replica-26 linear `depth4_diag`
trajectory sealed at 08:03:22 PDT, 62.76 minutes after the invocation began,
and finished with 0 unresolved fits out of 120. Pipeline execution and notebook
refreshes continue normally; 51.99 GiB remains free, and all 31 frozen hashes
still match. No scientific settings or stopping rules changed.

**Progress, 11:07 PDT October 16:** Phase 5 has **2,025 of 2,304** distinct
trajectories sealed (279 left), with 28 paired replicas complete per condition
and replica 28 progressing through linear rotation. The unchanged writer and
disk supervisor remain active without pipeline errors. Free disk is 51.31 GiB;
all 31 frozen scientific hashes still match. Continue the fixed final cohort,
closeout, and separate reporting-supplement integration.

**Progress, 15:10 PDT October 16:** all **31 paired replicas per condition**
(indices 0--30) are complete, with **2,232 of 2,304** distinct trajectories
sealed. Only the final 72 trajectories for replica 31 remain. The same
writer and disk supervisor continue without pipeline errors, the notebook
refreshed at 15:05:43 PDT, and 49.30 GiB is free. No scientific settings,
conditions, or stopping rules changed; closeout and final supplement follow.

**Historical continuation, 19:18 PDT October 15.** The user's new continuation request
supersedes the previous halt. Actual host preflight verified the lock free,
former PID 12071 absent, checkpoint and sealed inventory unchanged, all 31
frozen scientific hashes matching, and CUDA float64 working in `~/.venv`.
All **411 unit tests passed again in 86.20 seconds**. The tested reporting-only
driver now holds the actual writer lock (PID 6366, native session 90322);
independent disk supervisor session 60320 requests SIGTERM below 22 GiB free.
Free disk at launch was 52.49 GiB; preserve the 20-GiB reserve. Phase 5
continues the unchanged 32-replica cohort from **1,743 of 2,304 distinct
trajectories** (561 remain), followed by Phase 6 and final reporting-supplement
integration. Initial full validation and completed-unit replay still cost
time; actual end-to-end restart savings are not yet measured. No scientific
code, configuration, fit criteria, budgets, or stopping rules changed. Current
process identities, provenance, checkpoint, and restart instructions are in
[`akaike_score_gate/HANDOFF.md`](akaike_score_gate/HANDOFF.md).

**Progress, 20:51 PDT October 15:** Phase 5 has **1,758 of 2,304** distinct
trajectories sealed (546 left), with 5,159 sealed units overall and no
pipeline errors. The recovered replica-24 linear `asg_lr_lr` trajectory
sealed at 20:20:46 PDT, **62.46 minutes** after this invocation began.
This measures total startup/replay plus finishing the saved trajectory, not
a controlled reporting-only speedup. Its 4 unresolved fits out of 147 remain
included with health diagnostics; its interrupted calendar span is excluded
only from active-time extrapolation. Notebook refreshes continue; 52.40 GiB
remains free, and all 31 frozen hashes still match.

**Historical stop, 18:21 PDT October 15.** The user requested a graceful halt,
initially chose to wait, then explicitly authorized stopping the read-only
validator immediately after checkpoint preservation was reverified. SIGTERM
had been pending since 17:53:16 PDT; the inherited completed-unit replay
does not check its stop flag. The verified PID 12071 was terminated at
18:21:42 PDT (exit 137, not graceful exit 0). The disk supervisor exited 0.
Post-stop checks confirmed the lock free, worker absent, no GPU compute
processes, unchanged checkpoint bytes/timestamp and sealed-unit inventory,
and all 31 frozen scientific hashes matching. The node is available; **do
not resume without a new user request**. Phase 5 remains at 1,743 of 2,304
trajectories (561 left); Phase 6 and the final reporting supplement remain.
No new accepted scientific work occurred in this invocation. Free disk is
52.02 GiB. Root `running` status is stale; the notebook last refreshed at
18:13:15 PDT with zero errors. The incomplete invocation provenance is
preserved, not fabricated as complete. See the handoff for the exact restart
command and preserved checkpoint. The tested reporting-only driver remains
usable, but does not fix slow stop checks during archive replay; end-to-end
restart savings have not yet been measured.

**Historical continuation, 17:22 PDT October 15:** the approved reporting-only
resume driver was adopted after **411 unit tests passed** and all 31 frozen
scientific source hashes were verified unchanged. Its held-lock PID was
12071 (native exec session 95593); independent disk supervisor session 4418
used a stop threshold of 22 GiB free. Both have since exited as noted above.
Free disk was 52.03 GiB at cutover.
The old read-only replay process did not honor SIGTERM during completed-unit
reuse; it was terminated only after verifying unchanged checkpoint bytes,
timestamps, sealed inventory, source hashes, and historical staging. Its lock
release and exit were confirmed before starting the new driver. No accepted
scientific work was discarded. Initial/final integrity checks, normal unit
validation, and the 15-minute reporting heartbeat remain; measure actual
restart savings. Phase 5 remains at **1,743 of 2,304 distinct trajectories**,
with 561 left; Phase 6 and the final reporting supplement follow. Scientific
settings, cohort, budget accounting, and stopping rules are unchanged. Current
command, source hash, provenance location, and exact cutover details are in
[`akaike_score_gate/HANDOFF.md`](akaike_score_gate/HANDOFF.md).

**Historical restart, 16:00 PDT October 15:** Resumed at the user's request.
Preflight verified the actual lock was free, former PID 13471 absent, CUDA
float64 working in `~/.venv`, all 31 frozen scientific hashes matching,
and the resolved study hash unchanged. The new writer's held lock and exact
command are verified (actual PID 4306, exec session 19918); its independent
disk supervisor is session 57873. Do not launch a duplicate writer or trust
an old PID without checking its actual lock/command. Phases 0--4 are complete;
Phase 5 resumes from **1,743 of 2,304 distinct trajectories** sealed and the
checkpoint at replica 24 / linear / `asg_lr_lr`, step 33, stage `evaluate`,
depth 1. Replicas 0--23 are complete across all four conditions; replica 24
has 15 linear policies sealed. There are 561 final
trajectories left to seal. Configuration, cohort, and statistical stopping
rules remain unchanged. At preflight 54.05 GiB was free; preserve the
20-GiB reserve with the supervisor's 22-GiB graceful-stop threshold.
Progressive notebook refreshes continue after startup integrity/report replay,
which may take several hours without producing new trajectories. The latest
successful continuation took about 3 hours 51 minutes before its first new
seal; this slightly larger cohort also requires substantial block/report
replay. The preceding invocation ended during replay without adding a new
trajectory or advancing this checkpoint. The user approved investigation
of report-only resume overhead: redundant whole-report rebuilding occurs
after at least 48 reused Phase 2 audits and 96 completed Phase 5 condition
blocks before this checkpoint. Plot caching is already working. A separately
versioned wrapper could coalesce unchanged refreshes while preserving unit
validation, initial/final checks, and the 15-minute reporting heartbeat;
the user approved implementation and adoption only after equivalence and
artifact-safety tests pass. The separate candidate is being tested outside
the frozen package; the current worker and all
scientific sources remain unchanged. The checkpoint retains one unresolved fit with explicit health
diagnostics; it is not excluded from scientific results.
The notebook last refreshed at 12:24:51 PDT with 160 figures and no errors.
See the current continuation and restart notes in
[`akaike_score_gate/HANDOFF.md`](akaike_score_gate/HANDOFF.md).

**Historical progress, 04:13 PDT October 15:** Phase 5 had 1,689 sealed
trajectories and 5,086 units overall. The saved replica-23 sigmoid `depth1_lr` trajectory
completed at 03:59:42 PDT, about 3 hours 51 minutes after restart, retaining
120 fit calls and zero unresolved fits. Its recorded active online/evaluation
work is 23.56/41.40 seconds; exclude only its interrupted calendar span from
timing projections. The notebook refreshed at 03:59 PDT, 54.73 GiB remains
free, and no pipeline error is reported. The fixed cohort continues unchanged.

**Historical progress, 14:30 PDT October 14:** Phase 5 had 1,597 sealed
trajectories and 4,990 units overall, with replicas 0--21 complete across all
four conditions. No pipeline error was reported. The first new
trajectory after restart sealed at 05:14:48 PDT, about three hours after
launch; completed trajectories were verified, not retrained. The fixed
cohort continues unchanged, retaining unresolved optimizer states with
explicit health diagnostics.

**Historical progress, 22:51 PDT October 13:** Phase 5 had 1,019 sealed
trajectories and 4,380 units overall, with replicas 0--13 complete across all
four conditions. No pipeline error was reported.
The saved replica-12 `depth4_lr` trajectory completed at 20:00:58 PDT,
about 2 hours 42 minutes after restart, retaining 15 unresolved fits out
of 120. These accepted states remain in the results with explicit optimizer
health limitations. Recorded active online/evaluation work is 110.60/42.14
seconds; its interrupted calendar span is excluded only from timing
extrapolation. The fixed cohort continues unchanged.

**Historical progress, 13:15 PDT October 13:** Phase 5 had 723 sealed
trajectories and 4,068 units overall, with replicas 0--9 complete across all
four conditions. No execution errors were reported.
The saved sigmoid `depth2_diag` trajectory completed
at 09:24:45 PDT without unresolved fits, after approximately 2 hours 12 minutes
of startup validation/report replay. Its recorded active online/evaluation
times are 11.08/42.73 seconds; exclude only its interrupted calendar span
from compute extrapolation, not its scientific results. The fixed cohort
continues unchanged.

**Latest milestone, 12:29 PDT October 12:** Phase 4 completed at 12:16 PDT
with all 816 panel trajectories, 16 repair trajectories, four local repair
audits, profiling, and a frozen selection. Phase 5 started at 12:17 PDT;
its immutable ledger has 32 fresh paired replicas in each of four conditions,
2,432 nominal assignments and 2,304 distinct trajectories including the
promoted `repair_score128`. This repair retains LR EWC, a diagonal current
gate, $c=1$, half-life eight, cadence two, and rank eight, increasing only
current-score draws from 64 to 128. Promotion used the predeclared practical
tie tolerances, not a clear NLL improvement: its development mean NLL-AUC is
0.57153 versus 0.57072 for the baseline, and its mean active dimension is
56.125 versus 56.375. These are development diagnostics, not efficacy evidence.
Six fresh trajectories were sealed at that historical checkpoint. Main and
disk-supervisor sessions 96818/38630 exited during the 16:07 PDT pause;
the continuation above uses different, independently verified processes.

Phases 0--3 are complete. Phase 3 sealed all 960 trajectories, 768 local
shadows, 16 history-variance cases, and its final calibration decision.
Phase 4 began at 11:17 PDT on October 11 and resumed with 13 sealed
trajectories. Its saved replica-0 linear `depth6_lr` trajectory completed at
19:36 PDT without repeating previous fits or consolidations. By 03:19 PDT
on October 12, all 544 principal development trajectories were sealed:
eight replicas and 17 distinct policies in each of four conditions. The
272 additional scarcity/budget panel trajectories and the one triggered
repair have since completed. The selected settings are $c=1$,
history half-life eight arrivals, and gate cadence two arrivals. Comparator
aliases reduce the Phase 4 panels to 816 distinct trajectories before
bounded repairs, without changing nominal assignments or independent replica
counts. All 16 base variance-audit fits converged; 51 of 512 history refits
remain unresolved and are retained with explicit limitations. Phase 5 has
resumed; Phase 6 remains pending. These earlier diagnostics are development
and conditional research results, not fresh-cohort efficacy evidence.

At the historical October 11 pause, 2,448 units were sealed overall and
77.6 GiB remained free. The
notebook refreshed at 11:39:58 PDT, with 57 figure outputs and no error outputs.
A separately versioned artifact-only Phase 3 reporting extension is prepared
outside the frozen source set. Parent review and scalar-schema validation are
complete: all 365 Plan 14 unit tests pass, including 27 extension tests.
Actual scalar files were checksum-verified and plots inspected separately;
the identity guide no longer expands data ranges, and unresolved refits remain
explicit. Final notebook integration still waits for the experiment's last
refresh. No supplement was applied to the live notebook or completed runs.

All 31 frozen source hashes match; no scientific
code, configuration, or completed artifact was manually changed. See the
handoff for checkpoint-safe restart instructions and the substantial
verification/reporting overhead of resume. The previous continuation
added 157 trajectories; its first new seal followed about 2 hours 46 minutes
of completed-catalog replay. Timing caveats are documented below.

Historical milestone at 10:10 PDT on October 11: all 960 Phase 3 trajectories are
sealed, including all current-batch-only controls. Local shadow auditing
has started, with 27 cases sealed at this checkpoint. History-variance
audits and the final calibration decision still follow. This restart's
first new trajectory sealed at 08:15 PDT, about 3 hours 21 minutes after
launch. All 13 remaining trajectories were added without changing earlier
results; Phases 4--6 have not started.

Phase 0's
mathematical/timing validation and real-MNIST CPU/CUDA interruption checks
have completed, including 338 unit tests and an all-phase integration smoke.
The main scientific run resumed during Phase 1 after the agent handoff on
October 7, 2026, and temporary node reuse on October 8, most recently at
19:06 PDT. Phase 1 completed at 19:59 PDT on October 8: source/architecture
selection, paired assets, all three solver screens, family-common retention
selection, 112 paired driver assignments, and 48 local depth audits are sealed.
Phase 2 completed its 48 geometry and decision-sensitivity audits at 22:50
PDT on October 8, after approximately 2 hours 51 minutes. Phase 3's empirical
calibration completed on October 11; Phase 4 completed on October 12 and
the fresh Phase 5 resumed after node reuse on October 12. See
[`akaike_score_gate/HANDOFF.md`](akaike_score_gate/HANDOFF.md). Current status and
sealed results are recorded in `akaike_score_gate_results.ipynb`. The complete
default sequence proceeds without intermediate user approval, including
development, the bounded repair allowance, fresh evaluation, and progressive
reporting. Phase 0 smoke outcomes are pipeline checks, not efficacy evidence.

**Numerical preflight amendment:** five paired saved states exposed an
optimization plateau in the initial float32 prototype. With identical
starting parameters, evidence, anchors, and retention precision, only one
of five float32 fits met the $10^{-5}$ residual criterion at either tested
change tolerance. Float64 with change tolerance $10^{-12}$ met it in all
five. This is a numerical check, not efficacy evidence or a population
estimate. Freeze float64 model arithmetic and change tolerance $10^{-12}$
for every treatment and control, retaining the $10^{-5}$ residual criterion.
The initial $10^{-9}$ change tolerance below describes the starting proposal,
not the resolved execution setting. Reprofile all operation charges and
allowances on the resolved numerical configuration. The main run uses
`akaike_score_gate/configs/default_float64.json` and
`cache/mnist_experiment/akaike_score_gate/default_float64`; earlier preflight
artifacts remain separate and unchanged.

**Review amendment:** success is defined by useful learning under small-data
and small-compute budgets. Matching unrestricted full-network performance is
a secondary comparison, not a prerequisite. The added 4.4-expected-nine
condition is a bounded development study; the four-condition fresh cohort
remains unchanged in size.

**Scope:** Plan 14 studies how much of a pretrained network to activate. It
replaces SubGD adaptation geometry with nested layer restrictions and an
Akaike-inspired score gate, abbreviated ASG. It explicitly tests Fisher
representation quality in two different uses: the historical EWC penalty and
the gate's local prediction. It includes both rotation schedules and the
low-prevalence digit-9 mixture. It does not reopen optimal-$\pi$ research,
change completed artifacts, or presume that better Fisher estimation implies
better learning.

**Primary deliverables:** executable, resumable experiments under
`mnist_experiment/akaike_score_gate/`, immutable results under
`cache/mnist_experiment/akaike_score_gate/`, and an artifact-only
`mnist_experiment/akaike_score_gate_results.ipynb`. The theory below should be
self-contained enough for a later, evidence-qualified addition to
`mathematical_overview.ipynb`; that primer is not changed by this plan.

## Motivation And Questions

Plan 13 left practical head-only tuning competitive, while its learned SubGD
geometries did not establish a compelling general advantage. That motivates
structural restrictions whose meaning and operating cost are easier to
understand. It does not establish that one fixed restriction is always best.

The intended signal indicates when the current restriction has become
limiting and when another block would spend scarce information mainly on
estimation noise. A small active dimension is valuable insofar as it supports
useful learning within the available resources. Neither activating the full
network nor minimizing parameter count is an end in itself. Here statistical
efficiency means predictive performance obtained from new observations,
conditional on the shared pretrained model and historical information; it
does not claim attainment of a Cramer-Rao bound by the adaptive procedure.

The research questions are:

1. Does the useful adaptation depth vary across time, environments, or
   available evidence, and how large is the cost of choosing it incorrectly?
2. Can a local score calculation forecast the benefit of opening another
   layer after accounting for EWC and fitting optimism?
3. Does an exponentially weighted history improve those decisions relative
   to the newest batch alone?
4. Does retaining Fisher correlations materially improve EWC, the gate, both,
   or neither? Are cross-layer correlations particularly important?
5. At matched small data and compute budgets, does adaptive depth improve
   predictive learning relative to fixed depths, or reach useful quality with
   fewer new observations or less compute after counting all policy overhead?

Classical EWC uses a diagonal Fisher approximation [3]. Its use of a diagonal
matrix is not evidence that correlations cannot help a particular retained
objective. Conversely, lower Frobenius estimation error alone does not imply
a useful change in its penalty or in a Schur-complement solve. These are
separate empirical claims throughout the study.

An initially unsuccessful gate receives a planned diagnosis and a bounded
repair experiment. Poor efficacy is not an automatic stop. Architecture,
optimization, forecast accuracy, temporal lag, and probing overhead must be
examined before attributing failure to adaptive layer selection itself.

## Scientific Contract

### Likelihood, Fisher, And Distinct Roles

The loss is the negative log likelihood, with loss gradient and score related
by

$$
g(x;\theta)=\nabla_\theta L(x;\theta)=-s(x;\theta),
\qquad
\mathcal I(\theta)=\mathbb E_\theta[s(X;\theta)s(X;\theta)^T].
$$

There remains one likelihood Fisher field. Neither environmental prevalence
nor the choice of active layers defines a second Fisher estimand. Changing
prevalence moves the true parameter implicitly, as in the existing scientific
contract. Restrictions select principal coordinate blocks of the full model;
they do not redefine its likelihood.

Keep these objects separate in symbols, code, artifacts, and the glossary:

| Symbol | Meaning |
| --- | --- |
| $\mathcal I(\theta)$ | Unique unregularized likelihood Fisher |
| $\Omega_e$ | Historical EWC curvature summary entering evidence episode $e$ |
| $P_e$ | Actual quadratic retention precision in the mean-loss objective |
| $H_t$ | Current likelihood-curvature approximation used by the gate |
| $V_t$ | Sampling covariance of the weighted mean loss gradient |
| $K_t=H_t+P_e$ | Curvature of the local retained-objective approximation |
| $O_C$ | Estimated fitting optimism for active coordinate set $C$ |

$V_t$ is a noise covariance, not another Fisher definition. $\Omega_e$ is a
compressed historical approximation, not a claim that a fresh Fisher at the
current state exactly describes all old observations. Ridge changes the
estimator and its response; it does not manufacture likelihood information.

The original process chooses the parameter displacement. The auxiliary
process maintains the Fisher estimates and historical summary. This plan
explicitly uses direct observations and consolidation without LFU corrections
in its default conditions. This is an experimental omission of the full LFU,
not a claim that its residual term vanishes.

### Observation And Mean-Loss Units

Let $\mathcal W_t$ contain the observations in the current evidence episode.
For nonnegative weights summing to one, define

$$
L_t(\theta)=\sum_{i\in\mathcal W_t}a_{t,i}L(X_i;\theta),
\qquad
Q_t(\theta)=L_t(\theta)
+\frac12(\theta-\mu_e)^TP_e(\theta-\mu_e).
$$

The EWC anchor $\mu_e$ and precision $P_e$ are fixed during an episode. Every
depth comparison at a decision uses the same data, weights, anchor, penalty,
starting state, and optimization tolerances. The only changed coordinates
are the declared active set. Gate comparisons across Fisher treatments obey
the factorial contract below.

Use

$$
P_e=\omega(\Omega_e+\kappa_e I_d),
\qquad s_e=\frac{\operatorname{tr}(\Omega_e)}d.
$$

Starting values are $\omega=39$, $\kappa_e=.1s_e$ for rotations, and
$\omega=19$, $\kappa_e=0$ for digit-9 mixtures, motivated by Plans 12 and 13.
These are starting choices for a new architecture and buffering protocol.
The bounded Phase 1 retention check below may choose a shared alternative
before any gate calibration. The resolved $\omega$, $\kappa_e$, and effective
ridge coefficient $\omega\kappa_e$ must be recorded. They cannot be tuned
separately for diagonal and correlated Fisher conditions in the attribution
study.

All gradients, curvatures, and penalties here are in mean-observation units.
The mean-to-sum conversion is explicit in the classical derivation below.
Numerical solve damping is separately named and reported; it cannot silently
become statistical ridge or be used to claim a smaller effective dimension.

## Theory Of The Gate

The following is the proposed local risk approximation. Its algebraic
identities, statistical assumptions, and empirically calibrated components
must remain distinguishable in the notebook. Generalized information
criteria provide precedent for correcting the optimism of penalized
estimators [1], and penalized Fisher/Wilks theory explains why effective
dimension can replace ambient parameter count under suitable regularity [2].
Those results do not establish finite-sample guarantees for this entire
adaptive neural-network policy.

### Why Dimension Enters Predictive Risk

For an efficient regular MLE with per-observation Fisher $F$ in $d$ locally
identifiable coordinates,

$$
\operatorname{Cov}(\widehat\theta)\simeq\frac1nF^{-1},
\qquad n\operatorname{tr}\left[F\operatorname{Cov}(\widehat\theta)\right]
\simeq d.
$$

An omitted local displacement $b$ adds $nb^TFb$ to this Fisher-weighted
mean-squared-error approximation. A restriction trades estimation variance
against this displacement bias. On an identifiable restricted tangent space,
use its induced Fisher and dimension rather than the ambient dimension.

The quadratic expected test-loss excess from estimation is $d/(2n)$; the
training fit is optimistic by another $d/(2n)$ relative to the population
optimum. The training-to-test correction is therefore $d/n$, or $2d$ on the
summed negative twice-log-likelihood scale. This explains both the dimension
charge and the factor of two in the classical gate. The penalized case below
replaces this argument with the estimator's actual local response.

### Restricted Fit And Conditional Expansion

Let $A$ be the active suffix and $B$ the next frozen layer. All other
coordinates remain fixed at their current values. At the restricted fitted
state $\widehat\theta_A$, abbreviate

$$
g=\nabla L_t(\widehat\theta_A),
\qquad
r=g+P_e(\widehat\theta_A-\mu_e),
\qquad K=H+P_e.
$$

The condition for a solved restricted problem is $r_A\simeq0$, not $g_A=0$.
The candidate's penalty gradient includes cross-block effects caused by
movement in $A$. Ignoring these effects can make a likelihood-only score
recommend an inappropriate expansion.

The local retained-objective model is

$$
Q_t(\widehat\theta_A+\delta)
\simeq Q_t(\widehat\theta_A)+r^T\delta+\frac12\delta^TK\delta.
$$

With positive-definite $K$ on $A\cup B$, minimizing over the active
readjustment gives the Schur complement

$$
K_{B\mid A}=K_{BB}-K_{BA}K_{AA}^{-1}K_{AB}.
$$

At a converged restricted fit, the predicted expansion is

$$
\delta_B=-K_{B\mid A}^{-1}r_B,
\qquad
\delta_A=-K_{AA}^{-1}K_{AB}\delta_B.
$$

Thus the gate profiles over how the existing active layers can compensate
for the newly opened layer. It does not treat its gradient as an independent
source of improvement. The corresponding decrease in the retained quadratic
is $r_B^TK_{B\mid A}^{-1}r_B/2$.

For a nonzero restricted residual, the algebraically correct formulas use

$$
\widetilde r_B=r_B-K_{BA}K_{AA}^{-1}r_A,
\quad
\delta_B=-K_{B\mid A}^{-1}\widetilde r_B,
\quad
\delta_A=-K_{AA}^{-1}(r_A+K_{AB}\delta_B).
$$

These are diagnostic formulas, not permission to reward incomplete training
of $A$. An unresolved restricted optimum first receives the shared optimizer
retry. If it remains unresolved, the production gate records
`restricted_fit_unresolved` and stays at the current depth for that decision.
Any diagnostic improvement must subtract the active-only correction as well.

### Forecasting Data-Likelihood Gain

The gate targets improvement in predictive NLL. The proposed update is chosen
under retention, but the forecast evaluates the data term:

$$
\widehat G_B=-g^T\delta-\frac12\delta^TH\delta.
$$

This generally differs from the decrease in the retained quadratic. Releasing
penalty tension must not be mistaken for better predictions.

Use the likelihood Fisher/generalized Gauss-Newton matrix as the initial PSD
approximation to $H$. In a nonlinear network it need not equal the observed
NLL Hessian. At development checkpoints, compare predicted gain with both
the actual weighted NLL change at the preview point and the result of a
converged expanded shadow fit. Observed-Hessian products are diagnostic;
never materialize a full per-sample Hessian.

A proposal outside the calibrated local range may be shortened by a scalar
$\alpha\in\{1,1/2,1/4,\ldots,1/64\}$, selected by retained-objective
backtracking on the same available data. Evaluate $\widehat G_B$ using the
actual shortened displacement, and record $\alpha$. This adds a forward-pass
cost and another approximation to calibrate. It cannot receive an oracle
holdout line search. The unshortened case is the principal theoretical
derivation; frequent shortening is a reportable limitation.

### Fitting Optimism And Effective Dimension

Condition on the past archive and a locally fixed coordinate set $C$. Let
$\xi$ be the fluctuation of the weighted empirical loss gradient around its
population counterpart, with covariance $V$. The first-order estimator
response is

$$
\widehat\theta_C-\theta_C^{\mathrm{pop}}
\simeq-K_{CC}^{-1}\xi_C.
$$

The empirical loss contains the correlated term
$\xi_C^T(\widehat\theta_C-\theta_C^{\mathrm{pop}})$. Its expectation is
$-\operatorname{tr}(K_{CC}^{-1}V_{CC})$. An independent evaluation sample
does not share that term. This motivates the mean-NLL optimism estimate

$$
O_C=\operatorname{tr}(K_{CC}^{-1}V_{CC}),
\qquad
\Delta O_B=O_{A\cup B}-O_A.
$$

This derivation conditions on a fixed historical penalty. Uncertainty in
earlier training, selection among depths, nonlinear response, and temporal
drift require the subsequent trajectory-level calibration and evaluation.
They are not automatically corrected by this one trace.

Under independent, locally identically distributed observations,

$$
V=\left(\sum_i a_i^2\right)\Sigma_g,
\qquad
n_{\mathrm{eff}}=\frac1{\sum_i a_i^2}.
$$

Under the additional local information-equality approximation,
$\Sigma_g\simeq H\simeq\mathcal I$, giving

$$
O_C\simeq\frac{d_{\mathrm{eff},C}}{n_{\mathrm{eff}}},
\qquad
d_{\mathrm{eff},C}
=\operatorname{tr}\left[(H_{CC}+P_{CC})^{-1}H_{CC}\right].
$$

For positive-definite $P_{CC}$ and generalized eigenvalues satisfying
$H_{CC}v_j=\rho_jP_{CC}v_j$,

$$
d_{\mathrm{eff},C}=\sum_j\frac{\rho_j}{1+\rho_j}.
$$

Use limits on supported subspaces when appropriate. This is an optimism-based
effective dimension; it is not parameter variance itself. The latter's local
matrix approximation is $K_{CC}^{-1}V_{CC}K_{CC}^{-1}$.

With nonstationary independent samples the more general expression is
$V=\sum_i a_i^2\operatorname{Cov}(g_i)$; temporal dependence adds weighted
cross-covariances. The shortcut $H/n_{\mathrm{eff}}$ is a working approximation
whose calibration is explicitly tested. Model-generated score samples used
to improve a Fisher estimate do not increase the number of observed training
examples or $n_{\mathrm{eff}}$.

### Decision Rule And Classical Recovery

The calibrated local gate is

$$
\boxed{\widehat G_B>c\,\Delta O_B.}
$$

Equality stays at the smaller model. The starting value is $c=1$. The
multiplier $c$ is a development-calibrated optimism correction, not a
significance threshold or a separate penalty for every parameter. The
candidate grid is $c\in\{.5,1,2,4\}$. Keep the uncalibrated $c=1$ policy as
a named comparator.

Equivalently, the gate's forecast of predictive risk reduction is

$$
\widehat{\Delta R}_B=\widehat G_B-c\Delta O_B
\simeq\widehat G_B-c\frac{\Delta d_{\mathrm{eff},B}}{n_{\mathrm{eff}}}.
$$

Positive values favor expansion. Under the stated approximation, accumulated
information can reduce the estimation charge while persistent restriction
bias continues to supply a likelihood signal. Changes in the stream can
alter either term; selected depth need not increase monotonically with time.

Do not raise $c$ merely to force small models. Calibrate it for independent
predictive performance at the declared small-data and compute budgets. Keep
resource constraints explicit, including the cost of deciding whether to
expand. A resource-denied expansion is distinct from a negative statistical
gate. The $c=1$ classical limit remains a check of the risk calculation,
even when an external resource constraint prevents the resulting action.

For the classical limit, let $P=0$, $a_i=1/n$, $H=F$, $V=F/n$, and assume a
regular identifiable model and a converged restricted MLE. With summed
likelihood score $U=-ng$ and $q=|B|$,

$$
S_B=U_B^T(nF_{B\mid A})^{-1}U_B
=n\bar s_B^TF_{B\mid A}^{-1}\bar s_B,
\qquad
\widehat G_B=\frac{S_B}{2n},
\qquad
\Delta O_B=\frac qn.
$$

Consequently $c=1$ gives

$$
\boxed{\widehat G_B>\Delta O_B\quad\Longleftrightarrow\quad S_B>2q.}
$$

Under classical local alternatives, $S_B$ has an approximate noncentral
$\chi_q^2$ law with noncentrality
$\lambda_{\mathrm{nc}}=n\delta_\perp^TF_{B\mid A}\delta_\perp$ and mean
$q+\lambda_{\mathrm{nc}}$. The mean crosses the Akaike threshold when omitted
Fisher-weighted displacement exceeds the dimension cost. This is an
interpretation of the regular local model, not a finite-sample neural-network
power calculation or a guarantee about the probability of opening.

### Solve Geometry, Degeneracy, And Locality

All inverses in these formulas denote validated solves. Restrict the matrix
to $A\cup B$ before applying the block-inverse identity; inverting a larger
full-network matrix would also profile over layers that must stay frozen.
For $K=D+LL^T$, Woodbury yields

$$
K^{-1}v=D^{-1}v-D^{-1}L(I+L^TD^{-1}L)^{-1}L^TD^{-1}v.
$$

The low-rank-plus-diagonal factors of current curvature and retention can be
concatenated before a solve. Preserve the actual EWC precision while
approximating current curvature. Record solve residuals, minimum resolved
eigenvalues, condition estimates, and damping. A material negative
$\Delta O_B$ for common PSD inputs indicates inconsistent approximations or
a defect; it is not an invitation to open a layer for free.

Use a relative diagonal solve floor of at most $10^{-8}$ times the positive
mean diagonal in float64 as the numerical starting policy, with an absolute
machine-precision safeguard. Test tenfold floor perturbations in Phase 0.
If decisions change materially, classify them as unresolved and use the
shared statistical-ridge repair below. Do not advertise recovery of weak
directions through a numerical floor as observed information.

## Architecture And Nested Models

This is an explicit experiment-specific replacement of the canonical
512-parameter architecture requirement. Retain its two convolutional blocks,
SiLU activations, pooling, and 24-dimensional feature vector. Replace its
output head with

$$
24\xrightarrow{\mathrm{SiLU}}5
\xrightarrow{\mathrm{SiLU}}5
\xrightarrow{\mathrm{SiLU}}5
\longrightarrow9\text{ class contrasts}.
$$

The last affine map is followed by the existing fixed orthonormal map from
nine contrasts to ten sum-to-zero logits. It has no final SiLU. The three
hidden nonlinearities are required: a chain of affine maps alone would
collapse to one affine map and introduce continuous factorization
redundancies. Output gauge removal does not imply that every remaining
direction is statistically well identified.

| Depth | Active suffix | Added coordinates | Total active coordinates |
| --- | --- | ---: | ---: |
| 1 | Output contrast layer | 54 | 54 |
| 2 | Output plus last hidden dense layer | 30 | 84 |
| 3 | Output plus last two hidden dense layers | 30 | 114 |
| 4 | All four dense layers | 125 | 239 |
| 5 | Dense head plus second convolution | 222 | 461 |
| 6 | Entire network | 40 | 501 |

The complete width-five model has 501 trainable coordinates versus the old
gauge-fixed model's 487. Width six gives 559 and is the declared capacity
fallback. The table above is for width five; if width six is selected,
regenerate every depth count from its parameter layout. Store exact names,
shapes, coordinate offsets, active masks, and parameter counts. Never load
an old-model Fisher or checkpoint into the new
chart. Old code, datasets, and sampler implementations may be reused.

In the report use `output_only` for depth 1 and `dense_head` for depth 4.
The old Plan 13 `head_only` was a different 225-coordinate classifier; its
name must not imply a matched treatment in Plan 14.

Source-quality selection uses four development seeds, source-only validation,
and the old architecture as a diagnostic reference. Start at width five;
allow twice the original pretraining epoch budget if optimization has not
settled. Prefer width five when mean source NLL is within .02 nat and mean
accuracy within one percentage point of width six. Otherwise use width six.
If both remain substantially worse than the source reference, document the
capacity limitation and continue the best finite model through development.
Do not infer ASG efficacy from architecture-limited outcomes. Freeze width
before Phase 2; no confirmation-outcome-based architecture selection.

## Evidence Accumulation And Causal Scheduling

### Bounded Exponentially Weighted Episodes

Each episode holds at most 32 arrival batches. At an arrival, define

$$
b_{t,j}=2^{-(t-j)/h},
\qquad
a_{t,j,i}=\frac{b_{t,j}}{\sum_{k\in e}m_kb_{t,k}},
\quad i=1,\ldots,m_j.
$$

The half-life $h$ is measured in arrivals, with candidate values $4,8,16$.
Publish the corresponding observed-digit half-life for every condition.
All examples within one arrival have the same weight. The learner and gate
use the identical normalized weighted objective. The episode closes after
32 arrivals or at trajectory end; the final short episode is retained.
Buffer capacity is therefore bounded even in an indefinitely long stream.

For constant batch size $m$, decay $\beta=2^{-1/h}$, and an indefinitely long
locally stationary history,

$$
n_{\mathrm{eff}}\longrightarrow m\frac{1+\beta}{1-\beta}.
$$

For $m=4,h=8$ this limit is about 92.4 observations. The finite 32-arrival
episode and early warm-up have smaller effective sizes. Calculate actual
$n_{\mathrm{eff}}$ from recorded weights, never from the stationary limit.
Also record unique observations, current buffer size, expected and realized
digit-9 counts, and the age distribution of evidence. ESS is not a substitute
for those rare-event counts.

The prior anchor and precision are held fixed while an episode's observations
remain in the explicit objective. At episode end, using the final accepted
model state, estimate a full-coordinate weighted new Fisher $F_e^{\mathrm{new}}$
and perform one consolidation:

$$
\Omega_{e+1}=\mathcal C_E\left[(1-\eta_e)\Omega_e+
\eta_e F_e^{\mathrm{new}}\right],
\qquad
\mu_{e+1}=\theta_{\mathrm{end},e},
\qquad
\eta_e=1-(1-\pi)^{b_e}.
$$

Here $b_e$ is the number of arrivals in that episode, $\pi=1/(1+\omega)$,
and $\mathcal C_E$ is the declared EWC representation. Clear the buffer only
after checkpointing the consolidated state. No observation enters the
archive independently at every repeated fit. An observation-ID ledger must
verify this single-consolidation contract. New ridge is defined from the new
archive; old ridge is not accumulated as if it were likelihood information.

This mean-Fisher EMA is an explicitly approximate retention policy, not an
exact posterior precision recursion. Its episode cadence differs from Plan
13. Comparisons in Plan 14 are internally paired; cross-plan differences do
not isolate ASG. Keeping historical penalties coherent is particularly
important in EWC, where naive combinations can double-count evidence [4].

A current-batch-only ablation closes its episode after every arrival. Its
different retention cadence is reported as part of that complete protocol,
not attributed solely to ESS. An additional fixed-episode, newest-batch gate
diagnostic isolates information available to the gate without claiming that
it is a complete current-batch learner.

### Per-Arrival Order

1. Evaluate the incoming batch prequentially before its labels influence any
   update. Append it to the current buffer and record its exposure counters.
2. On a scheduled probe, start at depth 1 using the current parameter values;
   otherwise use the previous selected depth. Earlier layers are frozen,
   retaining their previously learned values.
3. Fit the restricted weighted objective. Use reset-per-solve strong-Wolfe
   coordinate L-BFGS and the common tolerances established in Phase 0.
4. At a scheduled probe, refresh the candidate score and local curvature,
   calculate the conditional preview and optimism difference, and apply the
   gate. An open gate triggers a real refit and tests the next layer.
5. A closed or unresolved gate stops expansion for this arrival. Later
   arrivals retry with their new data. If a probe is not scheduled, retain
   the previous selected depth for that arrival and fit that suffix.
6. Evaluate the accepted state on the independent current and retention
   panels. Consolidate only if the episode has ended, then seal the step.

Default probing is every arrival; candidates of every two or four arrivals
are calibrated for efficiency. Probing always occurs at episode start.
Within a probe there are at most five frontier tests and six fitted depths.
Use parameter warm starts between nested fits. Count all intermediate fits;
the final fit alone does not measure policy cost.

The starting solver budget is 50 L-BFGS iterations, at most 75 objective
evaluations, gradient tolerance $10^{-5}$, and change tolerance $10^{-9}$,
matching the established solver conventions. Permit one retry at twice the
budget on the same objective, recording its full cost. These are ceilings,
not a requirement to use every iteration. A stopping flag alone is not a
stationarity certificate: record the restricted residual norm and scaled
Newton decrement when available. Phase 0 freezes the numerical residual
criterion from finite-precision validation, and Phase 1 may raise the shared
budget to 100 or 200 iterations if more than 5% of development fits remain
unresolved. All matched depth controls receive the same rule. Production
cannot grant the gate an uncharged convergence oracle. Iteration ceilings
and retries remain subject to the shared adaptation allowance below; lack of
compute for convergence is recorded separately from a closed statistical gate.

Greedy stopping is explicitly temporary. Development shadows also test a
two-layer contiguous expansion at preassigned checkpoints where the first
gate closes. If consequential intermediate-layer blocking is observed,
the bounded repair condition may allow such a lookahead every eighth probe.
Its frequency and condition identity are frozen before fresh evaluation.

### Gradient And Curvature Cost

The mean score requires an aggregate backward pass through the active suffix
and candidate layer; the earlier prefix can remain detached for this probe.
Do not compute a backward pass separately for each layer. Cache frozen-prefix
features only with a parameter-state hash, and invalidate them when that
prefix changes.

The initial curvature budget is at most 64 input draws from the available
weighted buffer per probe and 256 at consolidation. Draws used only for
Monte Carlo integration do not become new training observations. Development
may promote a budget of 128 or 256 when the reference comparison shows a
material sampling limitation. Scalar and matrix accumulations use float64.

The likelihood Fisher is estimated through model-label score sampling, or
exact ten-class integration for a diagnostic/reference backend. Observed
training labels supply the loss gradient. Observed-label gradient covariance
may estimate $V$ but is not silently substituted for the model Fisher.
This explicit convention must be verified against existing helpers before
reuse; it is a declared Plan 14 estimator choice, not an assumption that all
older artifacts used this convention.

For softmax probabilities $p_\theta(x)$ and logit Jacobian $J_x$, the
model-label-integrated contribution is

$$
J_x^T\left[\operatorname{diag}(p_\theta(x))-
p_\theta(x)p_\theta(x)^T\right]J_x.
$$

Fisher-vector products can evaluate its action without storing a per-example
parameter Hessian. Dense matrices are affordable diagnostic instruments for
the approximately 500-coordinate model. A product solver and a dense solver
of the same sampled operator test numerical implementation, not two different
statistical estimators. Structured methods such as K-FAC [5] are a possible
later backend, not another default experimental axis.

Full-coordinate archive updates remain necessary even when a cheap gate
uses a suffix. Charge their cost to the method. Report learner fitting,
gate gradients, Fisher estimation, solves, consolidation, and reference-only
diagnostics separately. Frequent full-network diagnostic work is never
advertised as a deployable cheap policy.

## Fisher Accuracy: Attribution Design

### Two Factors With Different Jobs

The principal causal factorial has four cells:

| Condition | Historical EWC representation $E$ | Current gate representation $G$ |
| --- | --- | --- |
| `asg_diag_diag` | Diagonal | Diagonal |
| `asg_diag_lr` | Diagonal | Rank-eight plus residual diagonal |
| `asg_lr_diag` | Rank-eight plus residual diagonal | Diagonal |
| `asg_lr_lr` | Rank-eight plus residual diagonal | Rank-eight plus residual diagonal |

$E$ determines the actual objective precision $P_e$. $G$ approximates current
$H$ and the working $V=H/n_{\mathrm{eff}}$. Even a diagonal current gate must
retain the actual $P_e$ when forming $K$: `asg_lr_diag` can still have
cross-block curvature from EWC. Dropping that curvature would test a further
approximation to the objective and must be separately labeled.

Within each EWC row, the fixed-depth controls, including full space, provide
the corresponding performance and compute references. Comparisons of EWC rows
with a fixed depth isolate the retention representation without gate
selection; comparisons of gate columns at fixed EWC isolate current
geometry's role in selection. The four-cell interaction asks whether their
combination has an effect beyond those separate changes.

Hold the shared optimism multiplier, half-life, probe cadence, sampling
budget, retention strengths, and solver policy fixed across the four cells
in the principal attribution cohort. Different paths after treatment are
legitimate consequences; each condition estimates geometry at its own
parameter state. No condition may borrow a Fisher from another path.

At fixed states, independently vary historical representation while holding
current curvature at the reference, and current representation while holding
the actual EWC objective fixed. Changing EWC requires a new restricted fit
before evaluating a gate; reusing its old stationary point is invalid.

### Same-Sample And High-Sample References

Build these representations from one shared PSD score bank at each audit
state:

| Representation | Purpose |
| --- | --- |
| `diagonal` | Coordinate importance without off-diagonal entries |
| `layer_block` | Exact within-layer blocks, with cross-layer entries zeroed |
| `lr4_diag`, `lr8_diag`, `lr16_diag` | Increasing correlated rank with residual diagonal |
| `dense_same_samples` | Compression reference with the same sampling noise |
| `dense_reference` | High-sample estimate with an independently measured reference error |

The same-sample constructions must preserve the source diagonal and trace
up to a declared numerical tolerance. The low-rank residual diagonal is
nonnegative; report any clamp mass and diagonal discrepancy. A rank increase
is not assumed to improve decision quality monotonically.

The layer-block comparison matters: it distinguishes benefits of
within-layer covariance from benefits of the cross-layer terms in
$K_{BA}K_{AA}^{-1}K_{AB}$. As an additional offline attribution, compare
retained correlated factors with a coordinate-sign conjugation that preserves
PSD, eigenvalues, and diagonal while altering off-diagonal orientation. This
is a destructive geometry control in artifact analysis, not a production
recommendation or a guarantee that all useful orientation has been removed.

Start independent references at 4,096 inputs, with two independently seeded
halves. Expand to 8,192 if their relative Frobenius disagreement exceeds .05,
their relative conditional-solve disagreement exceeds .10 for a nonnegligible
residual, or they disagree on the reference gate. Normalize matrix/solve
differences by the mean of the two corresponding norms, with numerical zero
handled explicitly. If the expanded reference remains unresolved, retain
the state and label its comparison unresolved; do not expand indefinitely
or discard that state. Match the declared sampling convention,
parameter value, and temporal/covariate weighting. Exact label integration
removes model-label Monte Carlo noise but not finite-input uncertainty.
Also compare with the instantaneous current-environment reference to expose
lag from the recent history. Report that mismatch separately from compression
and same-target sampling error. Neither finite reference is called truth.

For historical precision, replay the saved consolidation history with dense
same-sample matrices. High-sample historical audits must estimate each saved
contribution at its own consolidation state. A fresh current Fisher cannot
replace this accumulated reference and be called an EWC estimation error.
Limit the high-sample history replay to the preassigned audit subset; it must
never supply undeclared extra data to production gates.

### Required Diagnostics

At each audit state record:

- Relative Frobenius and diagonal error, trace, leading subspace overlap,
  spectra, resolved rank, and condition estimates, with reference uncertainty.
- Cross-layer block errors and the size of the Schur subtraction relative to
  $K_{BB}$, including sensitivity to numerical floors.
- Conditional solve error for the actual residual, not just matrix error:
  compare $\widehat K_{B\mid A}^{-1}\widetilde r_B$ with its reference solve.
- Prediction error in $\widehat G_B$, $\Delta O_B$, selected depth, and the
  final open/closed decision, including disagreements near the threshold.
- EWC directional energy error along proposed and realized updates,
  $u^T(\widehat P-P^{\mathrm{ref}})u$, alongside actual retention NLL change.
- Gate regret: lost independent predictive gain from a missed opening or
  predictive degradation from an unnecessary opening, with uncertainty.
- Operator residuals and runtime for dense, low-rank, and product solvers of
  identical sampled matrices.

Report the full chain: improved matrix estimation, improved decision-relevant
quantities, changed decisions, and changed predictive/compute outcomes. A
break in that chain is an informative finding. In particular, Frobenius
error can be dominated by strong directions while inverse-sensitive gate
errors arise from weak directions or cancellation in the Schur complement.

## Environments, Pairing, And Outcomes

### Four Separately Reported Conditions

1. **Rotation, linear:** the established $0\to30\to0\to30$ degree path with
   40 transitions per arrow and four observations per arrival.
2. **Rotation, sigmoid:** the same endpoints, observations, and transitions
   with the established normalized-logistic schedule of sharpness eight.
3. **Digit-9 mixture, ten-positive scale:** $p\in\{0,.01,\ldots,.10\}$,
   18 observations per treatment arrival, following the common eight-digit
   $p=0$ update defined below. Treatment begins before the $.01$ arrival.
   Expected treatment exposure is
   $18\sum_{j=1}^{10}j/100=9.9$ digit-9 examples among 180 digits.
4. **Digit-9 mixture, hundred-positive scale:** the same schedule with 182
   observations per arrival, giving 100.1 expected digit-9 examples among
   1,820 treatment digits.

Keep linear and sigmoid as separate experimental conditions. Keep mixture
sample scales separate as well. No pooled result may conceal opposing
schedule or exposure effects. The mixture is a plausible shallow-adaptation
case, not an assertion that the output-only model must be optimal. Rotations
provide a complementary feature-shift setting. No performance-based choice
of a more favorable schedule is allowed.

Add a **development-only 4.4-expected-nine condition** using the same mixture
schedule and eight observations per treatment arrival: 80 treatment digits
and $8\sum_{j=1}^{10}j/100=4.4$ expected digit-9 observations. Its bounded
Phase 4 panel is specified below. It does not create a fifth full Fisher
audit or final-evaluation condition and supplies no confirmatory claim.

Source training uses the established upright all-digit initialization for
rotations and digits 0--8 for mixtures, with new model and data seeds. The
mixture receives no digit-9-rich burn-in. All three mixture exposure scales
share the same eight-observation $p=0$ full-space update, starting from the
same source model and common reference archive. Clone its post-update model
into the data-budget branches before assigning their Fisher treatments.
Thus the treatment-data ladder begins at a common learned function; different
warm-up sample counts cannot explain its effect. Regenerate the appropriate
initial Fisher at this common state before forming its paired representations.
Record the eight warm-up digits separately from treatment exposure. Rotation
treatment begins after source initialization; no lengthy adaptive calibration
prefix is discarded from the outcome. Preserve initialization, pre-treatment,
and post-treatment metrics, with explicit exposure origins.

Use four fixed, independently named seed families: architecture/protocol
development, gate calibration, causal development, and final evaluation.
Derive further named seeds for initialization, data order, reference panels,
model-label draws, randomized matrix algorithms, and bootstrap analysis.
Each paired replica shares initialization, the initial dense Fisher asset,
incoming observations and schedules, and evaluation panels. Different
representations derive from that common initial asset. Data and matrix seeds
remain independently named even when common random numbers pair treatments.

Separate source-validation, online-training, development-evaluation,
final-evaluation, and reference-estimation roles. The final evaluation panel
must not select architecture, pretraining epochs, retention, gate settings,
or repair policies. Freeze dataset partition IDs and all schedules before
their phase begins. Sample without replacement within the declared online
pool when possible; record duplicates and distinguish presentations from
unique observations if a sampler reuses examples.

Resolve all inherited sampler, initialization, holdout-size, and optimizer
settings into the Phase 0 configuration; no entry point may depend on an
unrecorded changing default. Each initial Fisher uses a 4,096-input dense
model-score asset shared by its paired treatments, with additional reference
work restricted to diagnostic branches. Initial Fisher observations are
disjoint from source fitting and online data. Their preparation cost and
any source-data assumptions are reported explicitly.

### Matched Data And Compute Budgets

For the mixture data ladder, hold the prevalence path, number of arrivals,
source model, common warm-up, and evaluation panels fixed while varying
observations per treatment arrival. Compare methods at matched $p$ positions
and matched accumulated exposure within each budget. Pair different exposure
scales through deterministic prefixes of per-arrival observation pools where
the existing sampler permits it; record the coupling and sampling convention.
Changing sample count must not silently change the path or class proportions.

A trajectory plot against cumulative digits alone mixes accumulating
information with changes in the environment. The matched-path data ladder
separates those effects more directly. Rotation curves describe learning
along their fixed changing tasks; compare policies at matching angles and
path positions, and do not infer a pure sample-size effect from progression
along one rotation trajectory.

Phase 0 defines two small, time-calibrated adaptation budgets for each
environment family using operation timings only. Profile a standard work
package consisting of an output-only fit, one frontier probe, one next-depth
fit, and amortized archive maintenance over preassigned buffer sizes. Let
$b_0$ be its measured upper-quartile cost. Use $b_0$ per arrival for the
strict development budget and $2b_0$ for the principal budget. Publish the
resolved absolute times and workload definition before gate development;
quality results cannot select these budgets. Mixture exposure scales share
one family's numeric budgets rather than receiving more compute simply
because their batches contain more data.

For reproducibility, implement the caps as deterministic allowances of
profiled work, with fixed charges for objective/gradient evaluations, score
sampling, matrix solves, and archive operations. The lookup includes active
depth, representation, and buffer size, and is frozen with the protocol.
Unused credit may carry forward; at arrival $t$ the policy cannot spend more
than $tb$ cumulative charged work or borrow from future arrivals. Report
actual synchronized wall time and discrepancies from the profiled allowance.
Equal charged budgets do not by themselves establish an actual speedup.

All policies, including each fixed depth and full space, receive the same
allowance. Count gate statistics and all intermediate fits. Reserve required
consolidation cost before admitting an expansion; a policy may retain its
last finite accepted state when the remaining allowance cannot support
another operation. Record budget-denied probes, incomplete fits, and any
unavoidable atomic-operation overrun. Never exempt ASG overhead or grant
full space additional iterations outside the budget. Only converged
restricted fits support the classical gate calculation; otherwise use the
previously declared unresolved-fit behavior.

The principal budget applies to the causal runs in Phases 3--5. The strict
budget is confined to the bounded Phase 4 mixture panel initially. Accurate
shadow/reference fits in Phases 1--3 may exceed these deployment budgets as
explicitly labeled research measurements; their performance is not that of
a deployable budget-matched control. This keeps the mathematical audit
separate from the effects of finite optimization resources.

### Fixed Depths And Statistical Efficiency

The complete Phase 4 and Phase 5 comparisons include all six fixed depths,
using the same weighted evidence and retained objective as ASG. Evaluate them
under both diagonal and rank-eight EWC. Include no update as boundary context.
The principal comparator is a fixed depth selected on development data under
the same resource constraints. Full space remains a useful capacity and
resource control; approaching its unconstrained performance is not a goal
that other useful outcomes must satisfy.

The principal learning endpoint is the across-replica mean of
exposure-normalized current-holdout NLL-AUC at the declared data and compute
budget. Show the full learning curve and the fixed early-exposure summaries,
so late performance cannot hide inefficient initial learning. Prequential
incoming-data NLL is a mandatory corroborating outcome. It is measured before
the batch trains the model and has no evaluation-panel sampling subsidy.
Report matched next-condition holdout gain for shadow calibration;
current-condition gain and drift are separate diagnostics.

Primary success is improved mean predictive learning at matched small
resource budgets, or reaching a predeclared useful quality level with fewer
new observations or less compute while maintaining retention. Report the
effects and their uncertainty; a tiny resolved effect is distinguished from
a practically important one. No universal MNIST quality threshold is assumed.
The earlier within-.01-nat/full-space and 20%-speedup comparison remains a
secondary descriptive row, not an eligibility rule or promotion target.

Retain a proposed source-retention allowance of .02 nat above the no-update
source reference, evaluated on the same source distribution, for development
selection. Publish effects over a range of retention tradeoffs as well. If no
policy satisfies the allowance, report that constraint failure and continue
the predeclared best-bet procedure; poor absolute performance is not rescued
by merely matching a poor full-space control.

| Metric | Role in the statistical-efficiency claim |
| --- | --- |
| Mean heldout NLL versus new observations at fixed compute | Predictive quality obtained from scarce data |
| NLL-AUC and early-exposure NLL-AUC | Learning over the whole path and its data-poor prefix |
| Observations and actual compute to fixed quality targets | Resources needed for useful performance |
| Fixed-prevalence precision/recall versus exposure | Interpretable rare-class learning |
| Active depth, effective dimension, and $d_{\mathrm{eff}}/n_{\mathrm{eff}}$ | How estimated capacity tracks evidence |
| Retention and depth-selection regret | Costs of forgetting, overfreezing, and unnecessary openings |

For each condition report NLL at 25%, 50%, and 100% of its treatment exposure,
using the first complete arrival at or beyond each fraction and stating its
actual count. Compute early AUC over the 25% and 50% prefixes, retaining
the same exposure normalization convention as the full-path endpoint. Freeze
the quadrature and treatment-exposure origin in the metric schema. Do not
select attractive intervals after seeing results or pool different path
positions as a pure sample-efficiency comparison.

Report data/time-to-target for illustrative heldout mean-NLL targets
$\{.5,.75,1.0\}$ nat, frozen before final evaluation and shared by all
methods. For the mixtures use the fixed-$p_{\mathrm{ref}}=.1$ evaluation
panel; for rotations use a fixed 30-degree panel. These threshold plots
measure quality on a fixed task while the training stream evolves and are
secondary to current-task NLL-AUC. Mark targets already satisfied at the
exposure origin, unreached targets, and later loss of an attained target.
Require attainment at two consecutive scheduled evaluations and count cost
through the second; report attainment fractions as well as censored exposure
and time. Do not average only successful replicas or extrapolate an
unobserved threshold crossing. Thresholds illustrate tradeoffs, not a claim
that any one of them defines application success.

Also report loss of predictive gain from missed beneficial openings and
degradation from unnecessary openings against independently scored local
depth alternatives. Distinguish statistical rejection from budget denial.
Effective-dimension ratios are health diagnostics, not an adequacy guarantee
or an outcome to minimize regardless of restriction bias. Publish the
quality/data/compute Pareto plots even when no single policy dominates.

Record mean current and retention accuracy, source/non-9 retention NLL,
worst predeclared retention stratum, calibration, parameter displacement,
optimization residuals, selected depth, ESS, effective dimension, gate
frequency, cost, and peak memory. A smaller active parameter count is not
itself a compute-saving result.

The principal adaptation timer includes online data handling, every
restricted/expanded fit, gate statistics, matrix solves, feature-cache
maintenance, and archive updates. Use synchronized CUDA timing at consistent
boundaries and paired execution ordering to limit load effects. Independent
holdout evaluation, diagnostic shadows, high-sample references, notebook
rendering, and deliberate pauses are separately reported research costs.
Report total experiment elapsed time as well, so neither a measurement
subsidy nor reporting overhead disappears from the accounting.

For every mixture scale, include digit-9 precision standardized to evaluation
prevalence $p_{\mathrm{ref}}=.1$, recall, false-positive rate, standardized
average precision, and threshold-swept precision-recall curves. Apply this
prevalence only to evaluation weighting. Show ordinary argmax operating
points; do not choose production thresholds using those curves. Store
confusion counts and undefined-precision flags so zero predicted positives
cannot silently disappear from averages. Publish the aggregation convention
and defined fraction beside mean precision.

Every mixture figure states its 4.4, 9.9, or 100.1 expected treatment count,
total digits, and realized-count summary. Mark the 4.4 evidence as development
only. Show precision and recall versus all integrated digits, with expected
cumulative 9s clearly distinguished from realized observations. Include
pretraining and common-update cost separately from adaptation cost.

## Phases And Autonomous Decisions

Phase outputs are immutable artifacts plus notebook sections in phase order.
The numbers below bound the default scope and support the timing estimate.
Development and final replica indices are disjoint. All listed environments
run even when an earlier environment looks disappointing.

### Phase 0: Mathematical, Numerical, And Execution Validation

Implement the model, coordinate restrictions, weighted objective, gate,
Fisher representations, artifact protocol, and minimal notebook. Validate:

- The 501/559 coordinate counts, ten-class probabilities, and fixed contrast
  map; affine-only collapse as an algebraic control.
- Score/loss-gradient signs; mean/sum scaling; Schur profiling versus a direct
  joint quadratic solve; and active-residual correction.
- Recovery of $S_B>2q$ at $c=1$ and zero regularization in an identifiable
  local likelihood example.
- Optimism and parameter covariance against finite replicated quadratic
  fits, including off-diagonal penalties and nonzero anchor displacement.
  These are estimator unit tests, not a revival of the retired Gaussian LFU
  experiment.
- Exact finite-weight ESS, squared weights in gradient-noise covariance,
  zero-information behavior, solve-floor sensitivity, and trace calculations.
- Fisher products against dense class-integrated matrices, PSD diagnostics
  before repair, and coordinate restriction before inversion.
- Buffer/anchor isolation, single consolidation, causal prequential timing,
  immutable completion, and interruption/resumption.
- Deterministic work charges, equal allowances across policies, denied-probe
  versus closed-gate status, and restoration of unspent credit on resume.
  Include archive reservations and intermediate fits in the accounting.

Run tiny CPU smoke configurations and one CUDA smoke for each environment
family. Use `~/.venv/bin/python`, confirm CUDA availability, and time complete
units including gate, evaluation, and archive work. An interrupted smoke must
match its uninterrupted counterpart under the documented reproducibility
tolerance. A second completed `--resume` must do zero scientific work.

Record the device and measured unit-time model in the notebook. No efficacy
screen or architecture selection is made from smoke outcomes. Numerical or
causal defects are repaired before producing interpretable development data.

### Phase 1: Architecture, Retention, And Depth Opportunity

Use four development replicas per condition. Run the source-only width
comparison and freeze the selected architecture. Establish six fixed-depth
trajectories and no update with the starting low-rank EWC, default $h=8$, and
the 32-arrival episode contract: 112 principal driver trajectories across
the four conditions. Preserve their checkpoints as audit inputs. These
convergence-oriented drivers measure depth opportunity; label them separately
from the deployment-budget controls in later phases.

Before freezing retention, run a bounded source/current tradeoff check on
the first two development replicas at depths 1, 4, and 6. Compare the starting
retention strength with factors $.5$ and $2$, and the starting ridge ratio
with $\kappa/s\in\{0,.01,.1\}$ where not already present. Vary strength and
ridge one axis at a time, not a Cartesian grid. Choose a common setting per
environment family by minimum mean current NLL, averaging the three tested
depths, subject to source-retention NLL no more than .02 nat above the
no-update source reference. Use this same retention reference in later policy
selection. Within .005 nat prefer the original setting. All mixture scales
share a retention setting, including the later 4.4 development panel. Linear
and sigmoid share one too. If none meet the margin, choose the minimum mean
current-plus-retention NLL and label the constraint failure. Freeze these
settings before Phase 2.

If this changes the starting setting, create new driver units for the final
setting; retain the earlier ones as development evidence. This bounded
replacement is included in the timing range. Every phase's config identifies
which driver generation it uses.

At preassigned early, middle, and late states, branch from a common state and
fit every depth on the same evidence. Separate this local comparison from
the six diverged fixed-depth trajectories. Estimate the opportunity for
depth selection and its regret, including uncertainty when depth differences
are small. A pointwise minimum of noisy holdout losses is an optimistic
hindsight benchmark, not an achievable policy. Use separate development
panels for choosing and scoring that local benchmark where feasible.

Continue even if one depth dominates: later work can still establish that
the gate recognizes this or that the proposed environment offers little
adaptive opportunity. Do not enlarge shift severity simply to manufacture
a favorable depth-switching story.

### Phase 2: Fisher Geometry And Decision Sensitivity

Use 48 preassigned audit cases: four replicas, four conditions, and three
trajectory positions. Balance source driver depths 1, 4, and 6 and candidate
frontiers 1 through 5 across cases before inspecting results. Each case stores
its pre-fit state, evidence buffer, actual archive, and restricted refits.

Produce the same-sample representation comparisons, high-sample current
references, and 16 independent low-budget score-bank realizations per case.
These replications measure estimator variability at fixed states. On eight
preassigned cases, also replay high-sample historical consolidation and
compare the full conditional expansions at all five frontiers.

Run the EWC-by-gate attribution and layer-block tests, with actual expanded
shadow fits and retention evaluations. Report bias/variance against finite
references, reference uncertainty, Schur error, optimism error, decision
disagreement, and realized predictive consequences. Rank-four and rank-16
sensitivities reuse score banks; they are not separate full trajectory
cohorts yet.

The output must answer whether errors are mainly sampling, compression,
missing cross-layer coupling, temporal mismatch, or inaccurate likelihood
curvature. If the dense same-sample gate fails too, rank inflation alone is
not an adequate repair. Do not pick an estimator solely by Frobenius error.

### Phase 3: Risk Calibration And Evidence Horizon

Use eight new development replicas per condition. Start with $h=8,c=1$ and
the four factorial cells. Include paired full-space controls under each EWC
representation for every distinct evidence protocol being compared; do not
borrow controls from different replicas or half-lives. At six predetermined
nonterminal positions per trajectory,
preserve local shadows for the next one- and two-layer expansions, restricted
objective residuals, preview predictions, and actual heldout improvements.
Additional gate events may be logged cheaply but do not replace these
preassigned audit positions.

Evaluate $c\in\{.5,1,2,4\}$ using stored local predictions where possible.
This reuses forecasts to assess calibration; it cannot replay an alternative
closed-loop trajectory whose parameters would have changed. Screen
$h\in\{4,8,16\}$ and probe intervals 1, 2, and 4 with small causal runs on
the first four replicas. Use coordinate-wise probing, retaining at most four
joint candidates for full eight-replica development. Include $c=1,h=8$ in
that shortlist regardless of initial rank.

On those first four replicas also run the complete current-batch-only
protocol at depths 1, 4, and 6 and with the $c=1$ low-rank/low-rank gate:
64 additional short-protocol trajectories across the four conditions.
Compare the complete buffering policies and, separately, the same-episode
newest-batch gate diagnostic defined above. These costs are included in the
Phase 3 range. Both comparisons are required even if the EMA policy initially
looks favorable.

Select common gate settings for the four Fisher cells by minimum mean
current-holdout NLL-AUC at the principal compute allowance and each condition's
fixed data budget. Keep those budgets constant across candidate settings.
Average the four Fisher cells within a condition, then average the four
conditions with equal weight; publish condition-specific results and
prequential NLL alongside this shared criterion. Apply the .02-nat source
retention allowance relative to no update to the same equal-weight average,
and disclose every condition-specific violation. The final assessment
remains condition-specific, not a pooled claim of success.

Within .005 nat of the best admissible candidate, prefer lower actual online
time; an unresolved timing tie prefers fewer mean active coordinates. Do not
increase $c$ merely to force freezing, or require closeness to full space.
If none satisfies retention, minimize current-plus-retention NLL and continue
as a failed-constraint development candidate. Budget-limited fits and denied
probes remain in the outcomes, not excluded failures. A violation of the
charged allowance requires an accounting fix, not a larger allowance chosen
from favorable outcomes. Freeze one shared policy for causal attribution.

Compare model-based $V=H/n_{\mathrm{eff}}$ with observed-label covariance and
repeated-history variance on 16 preassigned audit states. For the latter,
hold the archive and starting state fixed, draw 32 independent histories
from the declared schedule, and refit. This separates sampling variability
from changing means across time; a pooled covariance of drifting gradients
does not do that automatically. These are diagnostic replication units,
not 32 additional independent final-study replicas.

The notebook shows predicted versus observed gain, calibration by frontier
and ESS, optimism reliability, gate regret, rare-class counts, and opening
frequency, early-exposure learning, and capacity relative to evidence. The
empirical correction is judged on out-of-fold replicas; folds
never split time points from the same replica between fitting and scoring.
Because the first four replicas also determine the candidate shortlist, only
the remaining four serve as held-out scoring folds for that shortlist. Each
such fold excludes its entire replica from subsequent policy selection. The
screening replicas remain in development tables, not claimed as untouched
validation evidence.

### Phase 4: Causal Development And Bounded Repair

Use eight further replicas per condition for a common comparison set:
six fixed depths under each of two EWC representations, the four ASG cells,
the uncalibrated $c=1$ low-rank/low-rank gate, and no update. That is 18
conditions and 576 principal trajectories. Deduplicate identical conditions
by a shared immutable unit rather than fitting them twice.

This phase measures propagated effects and full cost, including all nested
fits and repeated probe work. It is the final development stage; it does not
supply final inferential evidence. Select the practical ASG cell and fixed
depth benchmarks before final-evaluation seeds are opened.

Add a bounded data/compute panel on the first four replicas: all 18 policies
at 4.4, 9.9, and 100.1 expected digit-9 observations under both strict and
principal allowances. This is 432 panel trajectories, of which 144 principal
9.9/100.1 trajectories already belong to the main comparison. Thus there are
288 additional trajectories and 864 total Phase 4 trajectories before any
repair. Pair source models, common warm-up, stream prefixes, and evaluation
panels across these branches as specified above.

Use the already selected gate settings and retention objective. This panel
does not add a hyperparameter grid, a full high-sample Fisher audit, or a
fresh-cohort condition. Its purpose is to expose overfreezing, high-variance
adaptation, and budget-denied learning as data and compute change separately.
Report matched-path NLL, standardized precision/recall, early learning, target
attainment, and active/effective dimension versus evidence. It may motivate
the existing bounded repair round, but remains development evidence.

Reserve one repair round of at most two additional policies, each on four
paired development replicas per condition, with the same controls. Choose
the repair from the measured failure mechanism:

| Observed issue | Authorized bounded repair |
| --- | --- |
| Current-curvature sampling dominates | Increase the probe score budget from 64 to 128 or 256 |
| Compression or cross-layer loss dominates | Promote rank 8 to 16, or use dense current curvature for this small network, with actual cost charged |
| Working gradient covariance miscalibrates optimism | Use the validated observed-score covariance estimate and recalibrate $c$ on development data |
| Large previews or poor restricted optimization | Tighten shared solve tolerances or use the validated shortened-preview rule |
| Evidence is stale | Use the shorter candidate half-life or more frequent probes |
| Intermediate layers repeatedly block useful joint moves | Add a two-layer lookahead every eighth probe |
| Gate overhead dominates | Use the validated two- or four-arrival cadence and state-hashed feature caching |
| Numerical null directions dominate | Apply a common $\kappa/s=.01$ statistical ridge to an explicitly new matched-control cohort |

All altered objectives require matched fixed-depth controls; they cannot
replace the original factorial cells. The original result remains in the
notebook. If a repair changes representation or covariance treatment, repeat
its local calibration checks before promotion. No repair can use final
evaluation panels, create more positive observations, discard inconvenient
replicas, or claim that model-generated labels are additional training data.
For a current-score-budget or rank repair, benchmark its complete probe at
the actual sample count and rank across the frozen buffer/frontier grid.
Replace only the applicable probe charges; preserve the original allowance
and all non-probe charges. Repeat one preassigned middle-position local
one-/two-layer shadow audit per condition on its first development replica,
and retain unresolved calibration outcomes alongside predictive results.

Promote at most one repaired policy as an extra final condition. If no repair
improves the declared development tradeoff, retain the original best bet and
complete final evaluation. The reservation provides a fair attempt, not an
unbounded hyperparameter search or an efficacy stop.

If the promoted repair changes the retained objective, EWC representation,
or fitting protocol, its final bundle must also include all six fixed depths
under that same change. Existing identical control units may be reused by
hash. A repair that only changes the gate's current-curvature estimate or
probe rule uses the existing matched controls. This distinction is resolved
in the freeze artifact before final work starts.

### Phase 5: Fresh Fixed-Cohort Evaluation

Freeze 32 new paired replicas for each of the four separately reported
conditions before any final result is inspected. Use the same 18 conditions
as Phase 4, plus at most one promoted repair: 2,304 nominal principal
trajectory assignments, 2,432 with a gate-only repair, or at most 3,200 when
six new matched fixed-depth controls are also required. Identical comparator
aliases reduce the currently selected principal protocol to 2,176 distinct
trajectories (2,304 with a gate-only repair), without reducing the 32 fresh
replicas in any experimental condition. Selected architecture, objective, half-life,
probe cadence, rank, score budget, optimism multiplier, principal adaptation
allowance, work-charge table, quality targets, and all fallbacks are immutable
in the final ledger. The strict-budget and 4.4 panels remain development-only.
This is a fixed cohort, not a claim of known power for an effect that has not
been measured yet.

Freeze one practical ASG policy and one best fixed-depth comparator per
condition using development results under the same data and compute budgets.
Report paired differences against that fixed comparator and budget-matched
full space. Use the declared NLL-AUC endpoint, with early-exposure curves,
retention, and resource-to-target results beside it. A useful outcome need
not match unrestricted full-space performance. Show all six depth controls,
but a favorable post hoc comparator choice is exploratory. Report both main
Fisher-factor effects and the interaction in the four-cell design using the
common shared calibration. A separately repaired condition is not silently
folded into that factorial estimand.

Use replica-level paired means and 95% intervals, with a seeded paired
bootstrap or the existing validated paired analysis. Time points, probe
events, and local resampling draws are not independent final replicas.
Report mean effects as primary and medians/win fractions as descriptive.
Use nominal condition-specific intervals without claiming simultaneous
coverage or a global winner selected across all conditions. The gate itself
does not conduct a sequential significance test.

No early stopping for significance and no extension until a desired result
appears. A user interruption yields a visibly incomplete fixed cohort with
descriptive results; it can later resume. Any subsequently authorized
replica extension gets a new frozen ledger and separately disclosed analysis.

Prioritize complete paired blocks, round-robin over the four conditions, so
an interrupted run leaves useful comparisons and progress plots for each.
Preserve every assigned replica, including rare-class-poor streams and
optimization failures. Report denominators and failure rates. Do not drop
failed treatments and present the survivors as the original cohort.

### Phase 6: Artifact Synthesis And Closeout

Rebuild and validate the notebook from immutable artifacts. Produce a
conclusion table with predictive gain, retention, adaptation time, memory,
data/time-to-target including nonattainment, selected depth, capacity relative
to evidence, gate reliability, and evidence status for each condition.
Answer the five motivating questions independently.

Distinguish the conclusions that more accurate geometry improves estimation,
improves local forecasts, changes decisions, improves learning, or improves
the practical compute tradeoff. Summarize null and adverse results with the
same visibility as positive ones. Include the extent to which development
calibration transferred to the fresh cohort.

Prepare a short theory-to-evidence note identifying which derivations are
algebraic identities, which approximations were empirically supported, and
which remain speculative. This supplies reviewable material for a later
primer amendment without presenting a successful calibration as a theorem.

## Progressive Notebook And Reporting

Create the notebook during Phase 0 and update it atomically after each
completed paired block or at least every 15 minutes while results accrue.
Every bounded invocation attempts a refresh on clean exit. An abrupt kill
may prevent the last refresh; a standalone artifact-only refresh command
must display all subsequently sealed units without resuming computation.

The first page reports phase, environment, planned/completed/failed units,
paired replica counts, elapsed active compute, pending work, timing forecast,
selected configuration hashes, and development/final status. Incomplete runs
are labeled as such. Failure to validate a purportedly complete required
artifact produces an explicit error, never silently missing evidence.

Sections follow phase order and include:

1. A reader's glossary and the gate equations, including mean/sum units,
   effective dimension, ESS, Schur profiling, and the $S_B>2q$ limit.
2. Architecture source quality, active-parameter table, optimizer health,
   fixed-depth trajectories, and the depth-opportunity audit.
3. Fisher spectra and error, layer-block comparisons, conditional solve error,
   EWC directional energy, and the two-factor attribution plots.
4. Forecast-versus-realized gain, optimism calibration, depth-versus-time,
   ESS, effective dimension relative to ESS, evidence ages, and
   greedy-frontier/blocked-lookahead diagnostics. Distinguish statistical
   rejection, inadequate optimization, and budget denial.
5. Current and retention NLL and accuracy, paired mean effects, prequential
   loss, early-exposure learning, matched data/compute panels, resource-to-target
   plots including censored outcomes, memory, and the complete cost breakdown.
6. Mixture precision/recall trajectories, standardized PR curves, and the
   statistical-efficiency plots labeled with 4.4, 9.9, or 100.1 expected 9s and
   all integrated-digit counts. Label the 4.4 and strict-budget panels as
   development-only. Report real and expected exposures separately and mark
   the common eight-digit warm-up outside treatment exposure.
7. A lab log of decisions, failed assumptions, repair triggers, completed
   repair results, and remaining questions as these become known.
8. The final comparison and conclusion tables, with unresolved claims stated.

Use confidence bands and consistent condition colors, label uncertainty as
pointwise or replica-level as appropriate, and export shareable PNG/SVG
figures under the run's report artifacts. Plot code must be reproducible from
recorded data. The notebook performs lightweight summaries and rendering
only: no training, data downloads, new score draws, large matrix estimation,
artifact repair, or automatic resumption. Cache expensive spectra, solves,
and bootstrap summaries in Python-produced analysis units.

## Immutable Runs And Unexpected Halts

Every unit lives under a unique protocol/configuration hash and records the
phase, condition, replica, architecture, sample identities, named seeds,
parameter layout, software/device/dtype versions, git commit and dirty
status, schema versions, parent hashes, budget tier and work-charge-table hash,
start/end times, and completion status. Source and resolved scientific
configuration hashes participate in compatibility checks. Completed artifacts
are append-only.

Use a deterministic manifest/ledger and explicit dependency graph. Write into
a temporary unit, flush all required data, validate checksums and invariants,
then atomically seal with `COMPLETED`. No consumer may interpret a partial
manifest as completion. Keep prior completion records even if a later code
version creates a replacement protocol.

All expensive runners support `--resume`, `--max-units`, and
`--max-wall-seconds`. The wall limit stops before starting further work and
checkpoints at safe boundaries; the current atomic GPU operation may finish
after the requested limit. Prevent duplicate writers with a run lock. A stale
lock may be reclaimed only after confirming its owner is gone; do not launch
duplicate work because a client re-sent a command.

Checkpoint after every accepted arrival and completed gate/refit boundary.
Persist parameters, archive and anchor, buffer IDs and weights, episode
position, current active mask, pending frontier, previous selected depth,
optimizer state when needed, generator states, counters, completed metric
rows, source-asset hashes, charged work, remaining credit, and budget-denial
state. Resume must not refill spent credit or double-charge a restored
accepted operation. Record replayed interrupted computation separately as
recovery overhead. Chunk expensive reference estimations into sealed partial
sums with sample ranges. Checkpoint long pretraining by epoch.

A power loss or `SIGKILL` can discard only the current unsealed chunk. Resume
validates and skips complete units, restores the last valid checkpoint, and
deterministically reruns the unfinished chunk without duplicating metric
rows or consolidating evidence twice. Do not depend on a signal handler to
save scientifically useful progress. Hardware-induced nondeterminism is
recorded and tested within numerical tolerances; identical bits across
different devices are not promised.

Test interruption during pretraining, local score estimation, a multi-depth
probe, a budget-denied operation, accepted-state saving, consolidation, and
report refresh. Verify completed checksums before and after resumption and
compare against an uninterrupted smoke. Corruption is quarantined and
reported, never overwritten as though the old completed artifact had been valid.

Large datasets, matrices, checkpoints, trajectories, logs, and report images
belong under ignored `cache/`. Source control contains implementation,
small configs, tests, this plan, and the progressively rendered notebook.

### Reporting-Only Resume Amendment

On October 15, the user authorized a separately versioned reporting-only
resume driver, conditional on equivalence and artifact-safety verification.
Repeated completed-block refreshes were consuming short compute windows:
the current restart revisits at least 48 Phase 2 audit callbacks and 96
completed Phase 5 condition-block callbacks before reaching new work.
Each callback rebuilds the whole report despite unchanged scientific units.
The existing image cache is working; increasing its size is not the remedy.

Keep all 31 frozen Python sources and the resolved scientific configuration
unchanged. The sibling module `akaike_score_gate_resume.py` delegates the
original CLI, scientific methods, unit validation, locks, and signal handling
to the original executor. It changes only redundant forced report requests:
when created-unit counts and nonvolatile progress are unchanged, use the
original timed refresh path instead. Progress JSON is still written on every
call, and the original 900-second reporting heartbeat remains active.

Initial, changed-data/status, missing-notebook, terminal, interrupted, and
error refreshes retain their normal behavior. Final forced checks are never
coalesced. A failed refresh must not advance the last-rendered identity.
Every reused unit still follows the original configuration, source-hash,
completion, and checksum validation path. No estimator, fit, gate, archive,
sampling, budget, replica, stopping, or uncertainty rule changes.

Restrict the driver to `--resume` of an existing exact frozen study with a
Phase 4 decision and final ledger. Record each invocation separately under
`execution/report_resume/`, outside scientific phase directories. Seal its
command, environment, exact driver source and SHA256, scientific source map,
initial/final inventories, outcome, and refresh counts. An invocation's
`COMPLETED` marker means that invocation record is immutable, not that the
scientific cohort is complete. A hard halt may leave its provenance staging
directory incomplete without invalidating sealed scientific units.

Before adoption, test baseline-versus-wrapper scientific dispatch and actual
tiny CPU trajectories, including fits, model tensors, gates, and charged
budgets. Test completed-run immutability, corruption detection, heartbeat,
terminal/error refreshes, failed-render retries, override restoration, and
provenance sealing. Run the existing unit suite and recheck all frozen
hashes. Switch the live worker only after those checks pass, recording the
cutover and recovery checkpoint. Initial full integrity scans and ordinary
unit verification still cost time; measure restart savings rather than
assuming instantaneous resumption.

**Verification and adoption, October 15, 17:22 PDT:** the combined suite passed
411 tests (365 existing and 46 new). Fixed-policy and adaptive LR/LR CPU
trajectories agree in model tensors, metrics, gate records, checkpoints, and
budget charges; completed reuse, corruption, random-state preservation,
heartbeat/final refreshes, and provenance are covered. All 31 frozen hashes
still match. Driver version 1 was adopted with the scientific checkpoint and
sealed inventory unchanged; see the current handoff for invocation provenance
and stop/cutover details. No scientific source or setting was changed.

## Execution Independence And Repair Boundaries

After authorization to execute Plan 14, the agent may implement and run all
six numbered research phases plus Phase 0, selecting the bounded development
choices above and reporting them immediately in the lab log. The user is not
a mandatory reviewer between phases. The final freeze artifact must exist
before any Phase 5 final-evaluation unit starts.

Poor NLL, few gate openings, budget-limited learning, or weak Fisher benefits
do not suspend the study. Continue the planned calibration, matched controls,
bounded repair, and fresh evaluation, labeling limitations. A repair cannot
silently replace old evidence or consume final outcomes as development data.

Hard blockers are mathematical normalization errors, broken data pairing,
future-label leakage, incompatible coordinate layouts, corrupt required
assets, or inability to define the planned objective. Repair an implementation
defect and rerun affected incomplete units under a new implementation version
when necessary. If safe repair is impossible, record the exact blocked
dependency and continue independent valid work; do not run invalid downstream
experiments merely to avoid an interruption.

A user request to stop takes effect immediately at the next safe boundary;
an unexpected machine halt is handled by resume. A hardware slowdown changes
the timing forecast, not sample counts or cohort composition. Once final
evaluation begins, optimization-health fallbacks remain those already frozen.

## Proposed Implementation And Commands

Reuse existing data/environment adapters, parameter layouts, contrast map,
coordinate L-BFGS, seeding, and artifact utilities where their contracts
match. Keep new model and episode logic in a Plan 14 namespace; do not mutate
the Plan 13 protocol to make its old artifacts look compatible.

Suggested modules are `config.py`, `model.py`, `evidence.py`, `gate.py`,
`fisher_audit.py`, `trajectory.py`, `calibration.py`, `analysis.py`,
`artifacts.py`, `orchestrate.py`, and `refresh_notebook.py`, with small
`configs/default.json`, `configs/smoke.json`, and `configs/cuda_smoke.json`.
Organize by actual responsibilities during implementation rather than adding
an abstraction for every mathematical symbol.

The implemented user interface is:

```bash
~/.venv/bin/python -m mnist_experiment.akaike_score_gate.orchestrate \
  --config mnist_experiment/akaike_score_gate/configs/default_float64.json \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 \
  --resume --max-wall-seconds 10800
```

```bash
~/.venv/bin/python -m mnist_experiment.akaike_score_gate.refresh_notebook \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 \
  --notebook mnist_experiment/akaike_score_gate_results.ipynb
```

The first invocation without a time limit runs the authorized frozen ledger
to completion; subsequent invocations reuse every valid completed unit.

After the reporting-only resume amendment has passed its verification,
already-frozen Phase 5 studies may use this separately recorded driver:

```bash
~/.venv/bin/python -m mnist_experiment.akaike_score_gate_resume \
  --config mnist_experiment/akaike_score_gate/configs/default_float64.json \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 \
  --notebook mnist_experiment/akaike_score_gate_results.ipynb \
  --resume
```

The same optional wall-time, unit, and phase limits retain the original CLI
semantics. The original entry point remains available; it does not silently
acquire this reporting optimization.

## Provisional Wall-Clock Compute Budget

**Observed execution note, October 7:** source/architecture selection took
about 15 active minutes. The first complete retention-pilot block uses one
replica, the starting retention, and the 50-iteration ceiling. Its measured
online-policy times are below; research evaluation, reporting, and resume
validation add elapsed time. These are individual timings, not population
means or estimates of deployable final-policy cost.

| Pilot depth | Linear rotation | Sigmoid rotation | Mixture, 9.9 expected 9s | Mixture, 100.1 expected 9s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 22.7 s | 21.6 s | 9.5 s | 14.7 s |
| 4 | 96.1 s | 99.1 s | 13.1 s | 17.6 s |
| 6 | 154.4 s | 152.8 s | 16.9 s | 72.6 s |

The complete 50-iteration pilot pass was sealed at approximately 02:32 UTC
on October 8 (October 7 in the client's timezone): 120 trajectories and
7,800 fits, of which 3,834 (49.15%) missed the unchanged $10^{-5}$ residual
certificate. This triggers the planned common 100-iteration pass. The resumed
invocation took about 3.1 hours to reach this milestone, predominantly this
pilot pass; the sealed trajectories record 2.20 hours of online-policy work
and 17.2 minutes of research evaluation. Resumption validation, setup,
checkpointing, and reporting also consume wall time. These development
results are not ASG efficacy evidence, and convergence does not substitute
for predictive performance.

At 04:44 UTC, the first 60 trajectories at 100 iterations were also sealed:
1,579 of 3,900 fits (40.49%) were unresolved, compared with 46.54% for the
same replica's 50-iteration pass. Their online-policy work totals 1.71 hours
and research evaluation 8.4 minutes; this half-pass took about 2.2 hours of
elapsed execution. Even if every remaining fit resolves, the complete
100-iteration grid will exceed the 5% threshold and require the planned
200-iteration pass. The entire assigned grid still runs; no pilots are
skipped on that basis.

The complete 100-iteration screen subsequently sealed all 120 trajectories:
3,256 of 7,800 fits (41.74%) were unresolved. Online-policy work totals
12,582.43 seconds (3.50 hours); research evaluation totals 1,014.56 seconds
(16.9 minutes). The 200-iteration screen is in progress. Calendar time spans
a user-directed node-reuse interruption, which is not active compute time.

The complete 200-iteration screen sealed all 120 trajectories: 2,500 of
7,800 fits (32.05%) were unresolved. Online-policy work totals 18,547.73
seconds (5.15 hours). The shared ceiling remains 200, with the unchanged
$10^{-5}$ certificate and an explicit unresolved-at-ceiling flag. Retention
selection keeps rotation's $\omega=39$ and ridge ratio $0.1$, and selects
mixture's $\omega=19$ and ridge ratio $0.01$. Both meet the pilot mean
source-retention constraint. These development choices do not establish ASG
efficacy; unresolved opportunities remain distinct from closed gates.

The original Phase 1 estimate of 1--3 hours is therefore replaced below by
13--20 hours. This extrapolates approximately 4.4 hours for the complete
100-iteration pass, then 5--8 hours for 200-iteration pilots and additional
driver/audit work. Phase 1 is now complete; its range below is the earlier
allocation, not remaining work. Keep the declared cohorts and convergence
criterion fixed.

Phase 2 completed in approximately 2 hours 51 minutes. Phase 3's first
162 sealed trajectories took approximately four hours. Calibration across
all four EWC-by-gate representation combinations is materially larger than
the original timing allowance:
512 adaptive screening trajectories and 96 distinct full-depth controls
across three evidence half-lives, up to 352 additional shortlist-confirmation
trajectories, and 64 current-batch-only trajectories. Thus up to 1,024 distinct
trajectories are followed by 768 baseline local-shadow audits and 16
history-variance cases with 32 refits each. These counts describe the existing
frozen design; they are not added experiments. Update Phase 3's forecast to
18--30 active hours, including research audits and reporting. Cadence and
opening frequency can change later unit rates, so this remains provisional.
Phases 4--6 still use unmeasured planning rates and must be updated as their
units complete. No cohort or statistical criterion is reduced.

**October 10 execution caveat:** the 18--30-hour Phase 3 allowance and the
subtotal ranges below are now outdated as forecasts. Candidate confirmation
has finished, but the current-batch-only ablation is materially slower than
the screened protocols. Its first complete replica across four conditions
took approximately four hours, suggesting roughly 16--20 hours for this
64-trajectory ablation alone at the observed rates, before the remaining
shadow and variance audits. A representative rotation trajectory used about
117 seconds of online-policy compute but took about 24 minutes to seal.
Snapshot histories, repeated checkpoint writes, and reporting are research
overhead, not deployable policy time. A completed current-batch rotation unit
stores approximately 3.6 GiB across its required artifacts. Preserve those
immutable outputs and monitor disk space. Repeated restart validation also
adds hours separately; the latest restart took about 2 hours 46 minutes to
its first new seal. Do not interpret the historical table as a remaining-time
promise. Publish a corrected whole-study forecast after later-phase rates
are measured; do not reduce cohorts or bypass compatibility checks to meet
the earlier allowance.

**Forward forecast, October 11:** before Phase 4 starts, allow roughly
6--16 active hours for its principal and scarcity/budget panels, plus any
triggered bounded repairs (earlier allowance 4--8 hours). The frozen scope
is unchanged: up to 864 distinct panel trajectories before repairs, with
legitimate comparator aliases potentially reducing that count. At the baseline
settings, sealed Phase 3 rotation units had median durations of about
113--156 seconds and mixture units about 9--20 seconds, depending on policy.
Those rates include saved research histories for adaptive drivers, whereas
routine Phase 4 paths disable snapshots. Other fixed depths and budget tiers
remain unmeasured here, so the new range is a planning extrapolation, not
measured Phase 4 runtime. Refine it after the first paired development
replicas. Reforecast Phase 5 before freezing its final execution ledger,
including its larger evaluation panel and all fresh paired replicas; its
earlier 6--14-hour allowance is superseded by the forecasts below.

**Measured development update, October 11:** the first complete Phase 4
replica has all 17 distinct policies in each of the four conditions. Mean
uninterrupted unit durations are about 79--83 seconds for rotations and
12--14 seconds for mixtures; the known paused trajectory is excluded from
wall-time extrapolation, not from scientific results. These rates suggest
8--12 active hours for all 816 Phase 4 panel trajectories, including paired
initialization and progressive reporting, before triggered bounded repairs.
Retain the earlier 4--8-hour repair allowance until its actual triggers/rates
are known. Refine these forecasts across the remaining development replicas.

For Phase 5, provisionally allow 35--50 active hours for its 2,176 distinct
principal trajectories, with up to roughly 2--4 additional hours if one
repair adds 128 assignments. Half the principal trajectories are rotations.
Their 10,000-image final panel is 2.5 times the development panel's size;
linear scaling of the measured roughly 17-second per-trajectory evaluation
component adds about 25 seconds per rotation trajectory. This is a cost
extrapolation, not a changed evaluation protocol or measured final-cohort
runtime. Include fresh source/asset creation, checkpointing, and reporting.
Update before the final execution ledger using all measured Phase 4 rates;
do not reduce replicas or alter stopping rules to meet the earlier estimate.

**Budget reconciliation, October 12:** the summary table below now uses
the revised Phase 4/5 forecasts instead of the obsolete 4--9/6--14-hour
allowances. All 544 principal development trajectories are sealed. Their
mean uninterrupted unit times are 76.22/77.99 seconds for linear/sigmoid
rotations and 12.31/10.67 seconds for the two mixture conditions. Extrapolating
the 10,000-image final panel gives approximately 31 hours of rotation
trajectories and four hours of mixture trajectories before fresh source/asset
creation and reporting allowances. Hence Phase 5's 35--50-hour allocation is
for the entire fresh computational cohort, not artifact-only analysis.
Phase 6 remains the separate, provisional 0.5--1-hour closeout. Source/asset
creation is included in the Phase 5 allowance; repeated restart validation
is additional overhead and depends on the number of interruptions.

Assume one otherwise available RTX 4070 using `~/.venv`, cached MNIST data,
and no concurrent GPU jobs. The budget is elapsed active experiment time,
not GPU-hours summed over devices and not calendar time during user pauses.
Per-phase ranges include ordinary analysis and progressive report refreshes.

Plan 13's Phase 6B completed 64 replicas and 320 treatment trajectories in
about 29 minutes, despite its earlier 5--7 hour forecast. Its short paths and
efficient batching make it an inadequate direct multiplier for this study.
Plan 14 adds up to 120 rotation transitions, repeated nested fits, weighted
buffers, and independent Fisher references. Those operations, especially
reference construction and calibration, dominate the uncertainty here.

The table distinguishes completed work from forward forecasts. Phase 4 is
now complete, including its one triggered repair. That repair and selection
took approximately 21 uninterrupted minutes, replacing its former 4--8-hour
provisional allowance as a forward estimate. Phase 5's whole-cohort range
includes the promoted repair; it is not a remaining-time stopwatch.
Completed-phase historical allocations must not be added to remaining work.
The former whole-study subtotal is withdrawn: Phase 3 exceeded its earlier
allocation, and its active time is not recoverable from calendar spans that
include node-reuse pauses.

| Phase | Scope and status at the 20:17 PDT October 12 preflight | Active wall-clock estimate |
| --- | --- | ---: |
| 0 | Math/contract tests, CPU and CUDA smokes, interruption drills, timing: complete | No remaining compute |
| 1 | Source checks, solver/retention pilots, principal drivers, replacements, and depth audits: complete | No remaining compute |
| 2 | Completed: 48 matrix/decision audit cases, sampling replications, eight history audits | 2.9 h measured |
| 3 | Complete: 960 distinct trajectories, 768 local shadows, 16 history-variance cases, and calibration decision | No remaining compute; old 18--30 h forecast exceeded |
| 4 | Complete: 816 distinct panel trajectories, 16 repair trajectories, four repair audits, and selection | No remaining compute; one repair/selection took 0.34 h |
| 5 | Resuming: 177 of 2,304 distinct trajectories sealed; 32 fresh replicas per condition, including the promoted repair | 37--54 h for the whole frozen cohort, not remaining time |
| 6 | Final integrity audit, report, and theory-to-evidence synthesis | 0.5--1 h |
| **Phase 5--6 subtotal** | **Whole frozen fresh cohort and closeout; excludes repeated restart validation** | **37.5--55 h** |
| **Promoted gate-only repair component** | **128 fresh trajectories; already included in Phase 5 above** | **2--4 h within that total, not extra again** |

The statistical-efficiency amendment's bounded Phase 4 panel is already
included in the revised 8--12-hour panel forecast; do not add its former
1--2-hour incremental allowance again. It does not multiply the fresh cohort
or the high-sample matrix audits. The reporting allowance covers the added
artifact-only learning and target-attainment summaries.

### Storage Forecast And Reserve

Read-only inspection on October 12 found approximately 72 GiB available
and 506 GiB in the current immutable study. Phase 2 occupies 178 GiB and
Phase 3 281 GiB; their large matrix/calibration audits are complete and are
not repeated in the remaining routine trajectories. Phase 4 trajectories
average approximately 6.2 MiB, including retained checkpoints. Phase 5 uses
the same no-calibration-snapshot protocol, with shared per-replica/family
assets and a larger evaluation panel. Holdout predictions are evaluated in
batches, not saved as an image-by-time tensor for every trajectory.

Allow approximately 20--22 GiB for Phase 5, including its fresh source/assets,
and 25--30 GiB conservatively for all remaining work, possible bounded repairs,
reports, and temporary writes. This leaves approximately 42--47 GiB if no
unrelated job consumes space. These are extrapolations, not hard upper bounds.
Preserve a 20-GiB free-space reserve and check growth during execution and
before phase transitions. The execution supervisor must request a graceful
stop before exhausting that reserve; the frozen Python runner does not itself
implement a disk-space guard. Never delete or alter sealed artifacts to
recover space. If growth exceeds the projection, pause and reassess storage
without reducing the cohort or changing scientific settings.

Allow roughly another 4--8 hours for initial implementation and engineering
work that is not active experimental compute. These are planning estimates,
not measured Plan 14 runtimes or guaranteed upper bounds. No substantial
compute is requested merely to refine this draft estimate.

Phase 0 replaces the provisional rates with measured medians and upper-tail
unit times. Before each later phase, publish a forecast from the frozen unit
count, observed rates, reference expansion counts, and reporting overhead.
Report implementation time, online-policy compute, research-only shadows,
reference estimation, and idle/paused time separately. Do not quietly shrink
the promised factorial or final cohort to fit an optimistic estimate.

## References And Theory Provenance

The gate derivation and proposed experimental design above are this
project's working development, not an assertion that the cited papers
establish this exact algorithm. The user-supplied ASG handoff motivates the
classical score/Akaike connection; the retained-objective and calibration
extensions are explicit additions.

1. Konishi, S. and Kitagawa, G. (1996), [Generalised information criteria in
   model selection](https://bemlar.ism.ac.jp/zhuang/Refs/Refs/kitagawa1996biometrika.pdf).
   General information-criterion corrections for estimators including
   penalized likelihood provide the relevant background.
2. Spokoiny, V., [Penalized maximum likelihood estimation and effective
   dimension](https://arxiv.org/abs/1205.0498). Penalized Fisher/Wilks theory
   motivates examining effective dimension and approximation conditions.
3. Kirkpatrick et al. (2017), [Overcoming catastrophic forgetting in neural
   networks](https://www.pnas.org/doi/10.1073/pnas.1611835114). The original EWC
   construction uses a diagonal Fisher precision approximation.
4. Huszar, F., [On Quadratic Penalties in Elastic Weight
   Consolidation](https://arxiv.org/abs/1712.03847). The sequential-penalty
   discussion motivates explicit evidence and anchor accounting.
5. Martens, J. and Grosse, R. (2015), [Optimizing Neural Networks with
   Kronecker-factored Approximate
   Curvature](https://proceedings.mlr.press/v37/martens15.html). A possible
   scalable curvature backend if the gate's decision sensitivity justifies
   further approximation work after this study.
