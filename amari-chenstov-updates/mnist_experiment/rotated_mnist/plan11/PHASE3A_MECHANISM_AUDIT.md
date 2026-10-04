# Plan 11 Phase 3A: Artifact-Only Mechanistic Audit

**Status:** Complete. This post hoc audit reads immutable Phase 2 and Phase 3
artifacts only. It does not train, evaluate a model, alter the Phase 3
estimands, or promote an exploratory endpoint to confirmatory evidence.

## Integrity And Scope

- Analysis contract: `plan11_phase3a_mechanism_audit_v1`.
- Analysis source:
  `mnist_experiment/rotated_mnist/plan11/mechanism_audit.py`.
- Analysis source SHA-256:
  `ccd555eec5c81ca665046abfa81325827bbb50ab9d35a5e584bb7499dfe90461`.
- Phase 2 study hash:
  `12b1ff1dbf3d0abdf4534cceaaacf738ab658de59d08286c2147131ae8ec645a`.
- Phase 3 study hash:
  `ffe20551118e39a405f02f31caa2765fc7075b6a116c785663b699a7e3e1fc98`.
- Validated Phase 2 inputs: 300 trajectories and 12 shared assets.
- Validated Phase 3 inputs: 1,408 trajectories and 352 shared assets.
- Every frozen contract, ledger, completed-file integrity hash, pairing check,
  action reconstruction, and terminal NLL/accuracy AUC reconstruction passed.

The favorable NLL contrast remains fixed-minus-blend, so positive values
favor $(c,\lambda)=(.025,.025)$. Linear mean gain was $-0.06207$ (95% CI
$[-0.12382,-0.00032]$, $p=.04884$); sigmoid mean gain was $-0.01966$
(95% CI $[-0.09035,0.05104]$, $p=.58487$). Phase 3 therefore remains a
negative confirmation for adaptive benefit.

## Observations

### Predictive Shape

The linear NLL disadvantage is not confined to one traversal leg. Mean
fixed-minus-blend gains were $-0.08581$ on the first ascent, $-0.06420$ on
descent, and $-0.03620$ on the second ascent. The mean exposure-indexed NLL
gap was most adverse at 116 observations ($-0.25804$), and averaged
$-0.16216$ from 75 through 150 observations. Running NLL AUC persists because
it integrates this earlier separation; it is not independent evidence.

Sigmoid timing differs descriptively. Its first-ascent gain was $+0.03158$,
followed by $-0.04511$ on descent and $-0.04543$ on the second ascent. None of
those post hoc leg summaries supplies a new success criterion.

Accuracy does not mirror the linear NLL result. Mean accuracy-AUC gain was
$-0.00258$ on linear and $+0.00027$ on sigmoid, with both intervals spanning
zero. Brier and expected-calibration-error AUC moved slightly against the
blend on linear ($-0.00623$ and $-0.00417$), but their post hoc intervals also
included zero. The observed effect is therefore primarily probability-quality
loss, not a reliable change in argmax decisions.

### Subgroups And Classes

Linear digit-9 NLL gain was $+0.00858$, while non-9 NLL gain was $-0.07000$.
Digit-9 recall gain was $-0.00120$. Thus the whole-path NLL disadvantage is
not a digit-9-specific failure; it is concentrated descriptively in the much
larger non-9 component. Class-8 recall had the largest adverse linear mean
($-0.01556$), but this was selected after inspecting ten classes and is not a
confirmatory subgroup finding. The aggregated confusion matrices show no
single substitution pattern large enough to explain the NLL effect.

### Coupled Policy Mechanics

After cold start, the linear blend's mean applied action was $.02593$ versus
$.02500$ fixed, an increase of only $.00093$. That small increase was coupled
to all of the following mean changes:

| Linear diagnostic | Blend minus fixed |
| --- | ---: |
| EWC old-to-new odds | $-1.3872$ |
| $q_t$ | $+0.000124$ |
| Fisher-weighted displacement norm | $+0.00442$ |
| Euclidean displacement norm | $-0.00033$ |
| Previous Fisher trace | $+160.16$ |
| Candidate Fisher trace | $+161.43$ |
| Fresh Fisher trace | $+151.69$ |

Sigmoid shows the same action scale ($+.00096$) and EWC-odds change
($-1.4496$), but smaller Fisher-trace differences and no resolved whole-path
NLL effect. The treatment simultaneously changes EWC regularization, direct
Fisher refresh, $q_t$, future recommendations, and the learner path. These
artifacts cannot identify one of those channels as the cause.

### Endogenous Timing And Heterogeneity

Across replicas, larger mean action excess was associated with worse NLL-AUC
gain: Pearson correlations were $-.422$ on linear and $-.333$ on sigmoid
(Spearman $-.339$ and $-.315$). Aggregate lead/lag correlations were strongly
negative at several visually selected lags, reaching $-.922$ at six updates
on linear and $-.660$ at fifteen updates on sigmoid. These are endogenous,
time-trended associations between a feedback action and its own future path;
they are useful hypothesis generators, not causal or inferential estimates.

### Numerical Pairing

All 352 pairs in each schedule had identical actions for the eight cold-start
updates, but no randomized-Lanczos update seed matched between policies:
0 of 41,888 comparisons per schedule. Parameter hashes matched at evaluation
steps 0 and 1, then diverged in every pair at step 2, well before adaptive
actions began.

This mismatch added material pre-treatment dispersion. At release, the paired
NLL-difference standard deviation was $.436$ on linear and $.538$ on sigmoid;
the cold-start NLL-AUC standard deviations were $.121$ and $.133$. Their
correlations with post-cold NLL-AUC gain were nevertheless weak (Pearson
$.088$ linear and $.049$ sigmoid). Policy-specific numerical randomness is
therefore a credible noise source and pairing defect, but the present audit
does not support it as the principal explanation for the mean linear result.
Every prospective successor must use common numerical-randomization seeds
within a pair; Phase 3B already records that requirement.

### Development Reversal

The 12-replica $(.025,.025)$ development mean changed from $+0.13709$ to
$-0.06207$ in linear confirmation, a shift of $-0.19916$. Its sign count
changed from 6/12 to 152/352. Sigmoid changed from $+0.03572$ (8/12) to
$-0.01966$ (171/352). The development intervals were wide and both included
zero. Selection and winner's curse are the most economical explanation for
why the pilot appeared favorable and the independent study did not.

## Ranked Explanations

1. **Selection and winner's curse:** strongest support for the pilot-to-
   confirmation reversal. The development effects were noisy grid-selection
   statistics and did not replicate.
2. **Coupled action increase harms probability quality on linear paths:**
   moderate descriptive support. Larger action excess tracks worse NLL, and
   the small action change consistently alters EWC odds, Fisher refresh, and
   Fisher-weighted movement. The responsible channel is unidentified.
3. **Schedule timing and reversal interact with the policy:** moderate but
   exploratory support. Sigmoid's first ascent differs from its later legs,
   while linear is adverse throughout.
4. **Calibration-sensitive, non-9 probability changes explain the lack of an
   accuracy signal:** supported as a description of the endpoint difference,
   not as an independently identified mechanism.
5. **Policy-specific Lanczos randomness inflates paired variance:** directly
   supported as a design defect; weak cold-to-post associations argue against
   it being the dominant mean-effect explanation.
6. **A digit-9 or single-class mechanism:** little support. No class subset is
   promoted from this post hoc review.

## Prospective Discriminator

The smallest causal successor remains a fresh paired $2\times2$ intervention:
fix or adapt the EWC action crossed with fix or adapt the direct-Fisher-refresh
action. It must use common numerical randomness, preserve the predictable
decision clock, and predeclare how $q_t$ and future recommendations are
updated in each arm. This would distinguish EWC, Fisher-refresh, and
interaction channels. It requires a separate protocol and fresh replicas;
Phase 3A authorizes no new compute.

Phase 3B asks a different question about improving deliberately imperfect
fixed anchors. It should retain its separate estimands and must not use this
post hoc audit to select windows, classes, lags, or outcomes.
