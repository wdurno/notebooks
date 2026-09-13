# Plan 11: Phase 0--2 Findings

**Status:** Development complete; independent confirmation not authorized.
The frozen precision gate stopped the study before Phase 3. These results are
exploratory, not a treatment-effect claim.

## Provenance And Completion

- Resolved development study hash:
  `12b1ff1dbf3d0abdf4534cceaaacf738ab658de59d08286c2147131ae8ec645a`.
- Artifacts: `cache/mnist_experiment/rotated_mnist/plan11/`. The Phase 0
  benchmark, development linear/sigmoid ledgers, linear shortlist, and final
  development decision are frozen there. Phase 1's exact-source smoke is under
  `cache/mnist_experiment/rotated_mnist/plan11_final_smoke/`.
- Phase 0 completed 36 paired full-length benchmark trajectories. Mean time
  was 58.25 seconds per trajectory; the 300-trajectory development maximum
  projected to 4.85 hours and about 1.09 GB, excluding setup. The largest
  2,560-trajectory confirmation projected to 41.42 learner hours and about
  14.67 GB, before fresh replica setup.
- Phase 1 completed nine tiny CPU smoke trajectories, a no-op `--resume`,
  post-cold action tests, and interrupted-unit lifecycle tests. One separate
  full GPU benchmark-policy repeat reproduced all 121 parameter hashes and
  scientific rows. PyTorch still warns that its CUDA adaptive-pooling
  backward kernel lacks a deterministic guarantee.
- Phase 2 completed all 216 linear and 84 sigmoid trajectories, plus 12
  distinct fresh model fits and 20,000-score initial Fishers. All 300 runs
  passed artifact hash checks, finite/predictable-action checks, and exact
  blend reconstruction; the maximum recorded $q$ recursion error was
  $2.78\times10^{-17}$. No replica or policy was replaced. Trajectories used
  4.82 GPU hours; replica setup added about 130 seconds.

## Development Results

The linear shortlist nominated $(c,\lambda)=(.01,.025)$ and
$(.025,.025)$. Fixed $.01$ had the lowest mean linear NLL AUC of the three
fixed anchors:

| Linear condition | Mean NLL AUC | Mean accuracy AUC |
| --- | ---: | ---: |
| Fixed $.01$ | 0.9621 | 0.7511 |
| Blend $(.01,.025)$ | 0.9666 | 0.7515 |
| Fixed $.025$ | 1.8527 | 0.6638 |
| Fixed $.05$ | 2.8085 | 0.6117 |

Sigmoid stress vetoed $(.025,.025)$: its mean NLL AUC was 1.7578, versus
1.0500 for the best fixed sigmoid control. It did not veto $(.01,.025)$,
whose mean sigmoid NLL AUC was 1.0434. The final selected development rule
was therefore $(.01,.025)$, compared primarily with fixed $.01$.

For the favorable paired contrast **fixed $.01$ minus blend** on linear,
the mean NLL-AUC gain was $-0.00455$ across 12 replicas (6 positive), with
paired standard deviation $0.1242$. The descriptive 95% Student-$t$
interval was $[-0.0835,0.0744]$; it is not an independent confirmation
interval. On sigmoid, the analogous mean was $+0.00660$ (8 positive), with
descriptive interval $[-0.0796,0.0928]$. Mean per-leg linear NLL-AUC gains
were $-0.0129$, $-0.0119$, and $+0.0111$; sigmoid gains were $-0.0028$,
$+0.0109$, and $+0.0116$. None of these exploratory signs can replace the
frozen linear primary endpoint.

The selected blend stayed close to its anchor: its post-cold mean applied
$\pi$ was $0.01085$ on linear and $0.01098$ on sigmoid, while the corresponding
raw recommendation means were $0.04387$ and $0.04925$. In this development
sample the small gain produced little action separation from fixed $.01$.

## Precision Gate

The one-sided 90% upper bound on the selected policy's paired linear
standard deviation was $0.17446$. The frozen design equation for a
$0.02$ NLL-AUC effect, 5% two-sided error, and 80% power rounded up to
**608 fresh confirmation replicas**. At the Phase 0 trajectory rate, that
would require about **98.4 learner hours** for five policies on both
schedules, before fresh setup. It exceeds the predeclared cap of 256
replicas, so the study stopped as **not resolvable at this budget**.

This is neither evidence of an adaptive benefit nor a theorem that shrinkage
cannot help. Fixed $.01$ looks descriptively strong on these paths, but it
too has not received a separate independent confirmation. No Phase 3
trajectories were run.
