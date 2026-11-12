# MNIST Evidence Synthesis Through Plan 18

**Scope:** Completed MNIST studies through Plan 18 Phase 9E, as of November
11, 2026. This is a narrative evidence synthesis, not a pooled statistical
meta-analysis: tasks, controls, and outcomes differ too much for one combined
effect size. It introduces no new experiments or statistical analyses.

## Motivation And Main Finding

The objective is continual learning when new observations, replay capacity,
and compute are limited. Pretraining provides an initial capability; the
online learner must acquire changing information without unnecessarily
destroying what it already knows.

**Carefully structured memory substantially improved online sample efficiency
in our strongest experiments.** A small short-term memory (STM), regularized
with finite-difference KL and combined with bounded replay and structured
Elastic Weight Consolidation (EWC), produced the clearest gains. Linearized
Fisher updates (LFUs) and adaptive $\pi$ also produced useful results, but
their advantages were more conditional and did not reliably add together.

The practical outcome is a set of resource trade-offs, not one universally
superior update rule. The mathematical foundation is in
[the theory notebook](mathematical_overview.ipynb); the earlier state of the
evidence is preserved in [MNIST findings](mnist-findings.ipynb).

## 1. Compressed Memory Earned Its Place

The early experiments established that retaining approximate likelihood
information can materially improve small-data learning:

- EWC improved environmental-accuracy AUC by **10.8 percentage points**, with
  a 95% interval of **[9.0, 12.6]**, over current-batch-only learning across
  ten fresh paired replicas.
- Bounded replay followed by eviction into EWC improved environmental-accuracy
  AUC by **3.65 points [1.77, 5.52]** over the same small replay buffer alone.
  This was not an improvement on every class-specific outcome.
- The hybrid approached unbounded replay's environmental accuracy with
  **13.7 times less recorded memory state**, but it was not computationally
  cheaper on the small network. Mean learner times were 46.2 seconds for the
  hybrid, 8.9 for bounded replay, and 19.7 for unbounded replay.

The enduring lesson is that **recent observations and compressed older
information are complementary resources**. The useful lifecycle was current
observation, then exact replay, then EWC admission upon eviction. Active replay
observations were not simultaneously counted in the archive.

Evidence: [Plan 2](mnist_experiment/plan2.md),
[Plan 3](mnist_experiment/plan3.md), and
[deployment comparisons](mnist_experiment/deployment_results.ipynb).

## 2. Small STM Produced The Strongest Efficiency Results

The important advance was an age-weighted finite-difference KL penalty on
sensitivity to STM perturbations. It was not simply the addition of trainable
parameters: the relevant replay controls also had STM parameters, EWC, and
ridge. Selected policies could nevertheless differ in calibrated ridge and
other settings, so these are policy comparisons, not universally isolated
effects of one loss term.

| Context | Estimated novel samples saved, nominal 95% CI | Compute at matched prediction quality | Qualification |
| --- | ---: | ---: | --- |
| Scalar angle regression, Plan 15 Phase 8.1 | 8.6% [1.8%, 21.2%] | 1.54x | Exploratory crossing analysis |
| Six-output registration, Plan 15 Phase 10 | 14.8%; CI withheld | 2.08x | Descriptive inverse estimate; matched retention unresolved |
| Nine-digit registration, Plan 15 Phase 11 | 38.2% [30.5%, 50.9%] | 0.775x | Full- and matched-budget retention requirements supported |
| Head-rank-one STM, Plan 15 Phase 12 | **41.4% [34.4%, 44.0%]** | **0.685x** | Full- and matched-budget retention requirements supported |
| Fresh reproduction, Plan 18 Phase 8 | 60.4%; CI withheld | 0.473x | Full-budget retention supported; matched-budget qualification unavailable |

These comparisons are against the specified Replay/EWC reference within each
study, not one common control across all rows. A novel sample is an image or
image pair in the earlier tasks, and a **grid containing nine digit pairs**
in the grid tasks. The percentages must not be pooled across tasks.

**Additional work per observation sometimes paid for itself through needing
fewer observations.** The head-rank-one treatment cost about 13.5% more at equal
exposure, but about 31.5% less online learning time at matched prediction
quality. This is stronger than merely exchanging compute for data, although
it is not a test against spending the same compute on additional replay.

The well-qualified headline remains **41.4% savings**. The later 60.4% estimate
is encouraging reproduction evidence, but its ordinary inverse confidence
interval was withheld because 107 of 2,000 bootstrap draws already met the
reference target at initialization. Its positive average predictive-risk
effect is better supported than that particular savings estimate.

Evidence: [Plan 15 results](mnist_experiment/fisher_memory_results.ipynb),
sections "Learning At Comparable Retention Cost" through "STM Capacity And
Placement"; [Plan 15 design](mnist_experiment/plan15.md);
[Plan 18 results](mnist_experiment/plan18_results.ipynb), section
"Reproduction And Blur Dependence"; and
[reproduction protocol](mnist_experiment/plan18.md#phase-8-reproduction-and-blur-dependence).

### What The Efficiency Statistics Mean

**Budget-averaged risk asks how well the learner predicts throughout a limited
data budget. Sample equivalence asks how much data it needs to match a
reference's prediction quality.** These are different questions.

For novel-sample count $n$, let $R_m(n)$ denote unregularized held-out risk
for method $m$. The budget-averaged statistic is

$$
A_m(N)=\frac{1}{N}\sum_{n=1}^{N}R_m(n).
$$

A smaller $A_m(N)$ establishes better average prediction over those budgets;
it does not itself specify a percentage of samples saved. Later studies
estimate sample equivalence by finding the last downward crossing of the mean
risk curve that stays below the reference's endpoint target through the
observed horizon, without extrapolation. The earlier Plan 15 Phase 7E
classification analysis instead used exploratory linear fits.

**$S$ means sample savings for the compute spent, at matched prediction quality.**

$$
S=\frac{\text{fraction of novel samples saved}}
        {\text{matched-quality learning-compute multiplier}}.
$$

For Head r1, $S=0.414/0.685\approx0.605$. **That is 41.4% data savings, not
60.5%.** $S$ is a summary score, not a probability, a percentage, or a universal
economic utility; it can exceed one.

Whole paired trajectories, not correlated checkpoints, are resampled for
the later bootstrap intervals. Unsupported inversions are not silently
discarded. The implementation is available in the
[STM-capacity analysis](mnist_experiment/fisher_memory_stm_capacity/analysis.py)
and the [grid-registration analysis](mnist_experiment/fisher_memory_grid_registration/analysis.py).

## 3. Where STM Looks Promising

**Small STM was sufficient.** Increasing STM from 108 head-only parameters to
688 distributed parameters barely changed sample savings. The smallest
configuration achieved the best measured $S$. "Head-only" described STM
placement: **all 3,268 LTM parameters remained trainable**. This is not evidence
that freezing the entire backbone would preserve the result. See the
[Phase 12 settings and comparisons](mnist_experiment/fisher_memory_results.ipynb)
and [memory layouts](mnist_experiment/fisher_memory_stm_capacity/model.py).

**Richer predictive tasks were more favorable.** Scalar angle regression was
weak or inconclusive; multivariate registration was better; joint nine-digit
registration was strongest. Phase 10 also found a larger FD benefit on the
same output coordinate when trained with the full vector. This supports
meaningful multi-output tasks as promising contexts, but does not isolate
output dimension from model capacity, supervision, pretraining, or replay
resources. See [Phases 9--12](mnist_experiment/plan15.md) and their
[reported outcomes](mnist_experiment/fisher_memory_results.ipynb).

**Appearance change mattered.** Plan 18's paired blur/no-blur comparison found
a larger relative F1 advantage under blur: the policy-by-context risk
interaction was **-0.0318 [-0.0667, -0.00418] square pixels per landmark**.
Both training and evaluation appearance changed. This establishes context
dependence, not a uniquely identified mechanism or a pure training-blur
effect. See [the reproduction and context results](mnist_experiment/plan18_results.ipynb).

**Information transfer remains a hypothesis.** Predictive gains, small STM
size, Fisher diagnostics, and STM-removal tests do not establish that literal
STM-to-LTM transfer causes the gains. Ordinary regularization and changes in
how the model uses its capacity remain explanations. The
[Plan 15 mechanism diagnostics](mnist_experiment/fisher_memory_results.ipynb)
retain this distinction.

## 4. Correlated Curvature Can Be Valuable Without Being Dense

The early representation study reduced relative Frobenius error from **0.978
for diagonal to 0.198 at rank 4 and 0.104 at rank 8**. Rank 8 also nearly
reproduced the dense coupled trajectory with far less persistent state. The
direct numerical and coupled comparison was a one-replica pilot, not a
universally validated rank choice. See the
[representation notebook](mnist_experiment/representation_results.ipynb) and
[original rank-study design](mnist_experiment/plan1.md#phase-7-diagonal-and-low-rank-plus-diagonal-representations).

Later, rank-64-plus-diagonal F1 beat the corresponding diagonal F1 on both
acquisition and retention in **all four development replicas**. Mean novel
risk was 2.2227 versus 2.3275. This supports the representation in that
setting; it does not prove that Fisher-estimation accuracy is the cause, or
that off-diagonal structure matters only when STM is present. The broader
six-condition retention-calibration gate did not open, so that representation
comparison has no fresh confirmation. See
[Plan 18 Phase 9 and its extension](mnist_experiment/plan18.md#phase-9e-adaptive-pi-with-the-original-non-diagonal-f1).

For a low-rank-plus-diagonal archive,

$$
\widehat{\mathcal I}=LL^T+\operatorname{diag}(d),\qquad
\Delta^T\widehat{\mathcal I}\Delta
=\lVert L^T\Delta\rVert^2+\sum_i d_i\Delta_i^2.
$$

Applying this penalty costs $O(pr)$ for $p$ parameters and rank $r$, rather
than $O(p^2)$ for an arbitrary dense matrix. Our current exact-compression
implementation nevertheless materializes a matrix and performs a full
eigendecomposition. In the four development pairs, compression accounted for
about 328 of 491 mean online learning seconds for non-diagonal F1, versus
128 total seconds for diagonal F1. That observed 3.83x cost is not a lower
bound on the cost of retaining correlated curvature. See the
[compression implementation](mnist_experiment/fisher_memory_grid_registration/compression.py)
and [representation operations](src/representations.py).

**A little correlated curvature may be enough.** Smaller ranks and reliable
operator-based compression are well-motivated opportunities, not established
replacements for the successful recipe. Low Frobenius error alone does not
certify unchanged acquisition or retention. The early unreorthogonalized
Lanczos failures at higher requested ranks were implementation/regime
limitations, not evidence against Lanczos generally.

## 5. LFUs And Adaptive Pi Have Conditional Roles

| Research thread | What survived | Evidence |
| --- | --- | --- |
| Matrix-valued LFUs | The derivative identity survived numerical validation, but noisy estimated corrections accumulated damaging errors and frequent indefiniteness. | [Early tracking](mnist_experiment/results.ipynb), [LFU isolation](mnist_experiment/plan3.md) |
| Diagonal LFUs | Paired finite differences improved digit-9 recall by **1.45 points [0.05, 2.85]** for about **2.8% extra wall time**, but cost **3.02 points of old-class retention**. This is an acquisition trade-off, not retention-matched sample savings. | [Plan 16 results](mnist_experiment/plan16_results.ipynb), [design](mnist_experiment/plan16.md), [curvature implementation](mnist_experiment/diagonal_lfu/curvature.py) |
| Early adaptive weights | Smoothed and shrunk local-risk recommendations did not reliably improve closed-loop prediction. A 352-replica-per-schedule follow-up failed to confirm the favorable pilot. | [Plan 11 confirmation](mnist_experiment/rotated_mnist/plan11/PHASE3_FINDINGS.md), [mechanism audit](mnist_experiment/rotated_mnist/plan11/PHASE3A_MECHANISM_AUDIT.md) |
| Velocity Kalman pi | Modest current-distribution tracking gains appeared during sustained rotation ramps. Digit mixtures and the latest non-diagonal F1 extension did not favor it over fixed pi=.025 on acquisition. | [Plan 17 results](mnist_experiment/plan17_results.ipynb), [Phase 7 design](mnist_experiment/plan17.md#phase-7-discrepancy-informed-adaptation-during-sustained-drift), [Plan 18 results](mnist_experiment/plan18_results.ipynb) |
| Combined methods | One retention-matched broad-drift configuration benefited substantially, but other configurations reversed the outcome. Benefits were not reliably additive. | [Plan 18](mnist_experiment/plan18.md), [Phases 6, 7, and 9E results](mnist_experiment/plan18_results.ipynb) |

In Plan 17 Phase 7, velocity Kalman reduced current-law RMS error by about
0.013--0.014 pixels relative to fixed $\pi=.025$ on both rotation amplitudes,
with nominal intervals excluding zero and slightly lower online wall time.
It did not improve every endpoint. In Plan 18 Phase 9E, adding it to the
non-diagonal F1 recipe instead increased average novel risk by
**0.02183 [0.00546, 0.04275]**, improved endpoint old-data retention, and cost
about 5.3% more online compute. Ridge differed after calibration, so this
compares complete policies rather than adaptive timing alone.

The useful application hypothesis is **choosing adaptation weights from one
continuing stream when repeated learner-tuning trials are unavailable**.
Observer settings were still selected in development, and superiority to
every fixed weight or reliable transfer to unfamiliar streams is unproved.
The mathematical optimum is conditional on a local risk model, not a theorem
of globally optimal continual learning.

The favorable combined result also needs its qualifications: Plan 18 Phase 6
estimated **49.1% [3.0%, 50.3%]** outbound-grid savings at 1.33x matched
compute, with the specified retention checks supported. Four fresh replicas
and nonmonotone curves made the interval highly asymmetric. The result
described earlier sustained quality, not twice-as-fast first acquisition;
absolute forgetting remained substantial. It does not override the later
negative combination results at different operating points.

## 6. Restrictions And Regularizers Were Not Automatic Wins

**SubGD did not become the practical default.** Head-only tuning was attractive
with extremely few positive examples. At about 100 expected positive examples,
full-space learning became more competitive; adaptive SubGD with a floor
mostly matched it, with small secondary improvements rather than a clear
precision-recall advantage. See [Plan 13](mnist_experiment/plan13.md) and
[its results](mnist_experiment/continual_subgd_results.ipynb).

**Akaike Score Gating did not establish a primary predictive advantage.** Across
32 fresh replicas per condition, the selected adaptive policy lost to the
fixed control on mean NLL-AUC in three conditions; the fourth was unresolved.
Some policies reduced active parameters or online time, but those savings
came with predictive trade-offs. See [Plan 14](mnist_experiment/plan14.md) and
[the ASG conclusion](mnist_experiment/akaike_score_gate_results.ipynb).

**Stable parameters were not sufficient evidence of good learning.** Modest
isotropic ridge sometimes improved the acquisition-retention balance, whereas
the tested MP-based spectral penalties suppressed learning excessively.
There is no universally best ridge constant in these results. See
[Plan 12](mnist_experiment/plan12.md),
[ridge estimator health](mnist_experiment/rotated_mnist/ridge_estimator_health.ipynb),
and [the low-quantile follow-up](mnist_experiment/rotated_mnist/plan12/PHASE5_FINDINGS.md).

## 7. What This Means For Limited-Resource Applications

| Dominant constraint | Evidence-backed starting point | Important limit |
| --- | --- | --- |
| New observations are expensive | Small FD-KL STM, bounded replay, and correlated EWC for a pretrained multi-output predictor | Strongest evidence is the grid task; larger-model transfer remains untested |
| Online compute is very scarce | Benchmark simpler adaptation and inexpensive archive representations first | Fewer parameters or a diagonal archive need not preserve prediction quality |
| Replay storage is scarce | Retire observations into a compressed archive after a short exact replay period | Compression also consumes compute; it is not a free substitute for replay |
| Repeated tuning trials are unavailable | Consider the calibrated velocity-Kalman adaptation policy | Observer settings still need to transfer; fixed pi remains a strong control |

We have not established an operationally optimal replay/EWC allocation or
shown that the winning STM policy beats spending identical compute on more
replay. These are remaining resource-allocation questions, not reasons to
discard the demonstrated gains at the tested allocations.

## Interpretation Boundaries

- **Retention is part of the comparison.** Development calibration must be
  checked on fresh trajectories. A nonsignificant difference is not
  equivalence, and matching two retention summaries is not pointwise matching.
  Plan 18 Phase 5 explicitly failed to provide the intended fair comparison;
  its observations remain preserved, not pooled into favorable evidence.
  See [the calibration-failure conclusion](mnist_experiment/plan18.md#phase-5-conclusion-retention-calibration-failed).
- **Novel-sample savings are not total-data savings.** They condition on
  pretraining and development. At matched checkpoints, background exposure
  and drift history differ. A mean-curve crossing is not a causal estimate of
  deleting labels, nor the mean number each individual learner needs.
- **Online cost is not total research or deployment cost.** Pretraining,
  tuning, evaluation, recovery I/O, and hardware utilization matter. Some
  studies report learner time; others additionally report wall time. Their
  ratios cannot be transplanted unchanged to larger models.
- **Applied metrics must match the task.** NLL gains need not change argmax
  accuracy. Rare-class accuracy can hide a useless detector; precision and
  recall at a fixed evaluation prevalence were more informative. Registration
  error and landmark success made later gains more directly interpretable.
- **Intervals have limited scope.** Most later intervals are nominal paired
  bootstrap intervals, not familywise guarantees across all explored settings.
  Small cohorts, repeated use of MNIST, development selection, bounded solves,
  and unstable inverse crossings limit extrapolation. The independent
  replication steps matter, but they do not establish cross-domain validity.

## Conclusion

**Structured memory can make scarce new observations substantially more
useful, without a large STM or necessarily more online compute to reach a
useful prediction quality.** The strongest supported recipe used a tiny
head-rank-one STM, FD-KL regularization, bounded replay, and a correlated EWC
archive. Its retention-qualified 41.4% online sample saving is the clearest
headline; the fresh 60.4% point estimate strengthens interest without resolving
all of its uncertainty.

LFUs and adaptive pi remain context-dependent tools rather than mandatory
additions. The next applied challenge is to preserve the demonstrated benefit
while reducing archive-maintenance cost and transferring the calibrated
recipe beyond these small MNIST models.

### Reading The Evidence On GitHub

Links here are relative to the repository root and point to plans, saved
notebooks, findings notes, and source code. The notebooks contain the human
readable plots and tables; their sections are named above where useful.
Large immutable checkpoints and raw run artifacts live under ignored
`cache/` directories and are **not bundled on GitHub**. Links into those raw
artifacts inside individual notebooks require the corresponding local data.
This synthesis does not imply that every historical run can be reconstructed
from notebook outputs alone.
