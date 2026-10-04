# Plan 11: Focused Own-Anchor Phase 3 Findings

**Status:** Independent confirmation complete for $(c,\lambda)=(.025,.025)$
versus fixed $\pi=.025$ on the linear and sigmoid double-lap schedules.
This is the 2026-09-13 amendment to [Plan 11](../../plan11.md), not a
reopening of the original stopped best-fixed comparison. The exploratory
12-replica development data do not enter the intervals below.

## Completion And Provenance

- Frozen study hash:
  `ffe20551118e39a405f02f31caa2765fc7075b6a116c785663b699a7e3e1fc98`.
- Immutable artifacts:
  `cache/mnist_experiment/rotated_mnist/plan11_followup025_v1/`.
  The contract fixes 352 new replica identities, both schedules, and only
  fixed $.025$ and blend $(.025,.025)$.
- All 352 shared initializations/Fishers and all 1,408 policy trajectories
  completed. The artifact-only loader verified completed-file hashes,
  pairing, finite metrics, predictable actions, exact blend checks, exposure
  alignment, and reproduction of terminal NLL/accuracy AUCs from each stored
  trajectory. No blend fallback was recorded. A halt during replica 140's
  shared-asset setup was resumed under the original frozen ledger; no
  completed run was changed or replacement replica added. A final no-op
  `--resume` completed zero new pairs.
- Measured compute summed to 23.43 trajectory hours plus 1.10 hours of
  shared setup, or **24.52 hours**. Calendar elapsed time was longer because
  the machine was retasked between sessions. PyTorch continued to warn that
  its CUDA adaptive-pooling backward kernel is not guaranteed deterministic.

## Schedule-Specific Mean Effects

The favorable paired contrast is fixed-$.025$ NLL AUC minus blend NLL AUC.
Intervals are two-sided 95% Student-$t$ intervals over 352 independent
replicas **within each schedule**. The $p$-values are unadjusted and do not
support a familywise-5% claim that either schedule works.

| Schedule | Fixed NLL AUC | Blend NLL AUC | Mean favorable gain | 95% CI | Two-sided $p$ | Favorable replicas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 1.60899 | 1.67106 | $-0.06207$ | $[-0.12382,-0.00032]$ | $0.04884$ | 152/352 |
| Sigmoid | 1.79346 | 1.81311 | $-0.01966$ | $[-0.09035,0.05104]$ | $0.58487$ | 171/352 |

The linear point estimate favors **fixed** $.025$; its nominal interval only
barely excludes zero on the harmful side. Sigmoid is unresolved, with a
negative point estimate and an interval spanning both signs. Neither
schedule meets the predeclared criterion of positive mean NLL-AUC gain with
$p<.05$. This is no evidence for promoting the blend over its own anchor;
the narrow linear significance should not be overstated as a broad theorem
of harm or a claim against all possible locally adaptive policies.

As descriptive secondary context, mean environmental-accuracy AUC was
$.67761$ fixed versus $.67502$ blend on linear (paired blend-minus-fixed
gain $-.00258$, 95% interval $[-.00805,.00288]$) and $.65690$ fixed versus
$.65717$ blend on sigmoid (gain $+.00027$, interval $[-.00552,.00605]$).
The [artifact-only notebook](../shrunk_pi_results.ipynb) shows the mean
current-environment NLL and accuracy trajectories and their running AUCs
without changing the NLL-AUC primary endpoint.

The $.01$ same-anchor backup was **not** launched. Any decision to study it
requires a separately agreed, versioned protocol and fresh confirmation
data, as specified in the amendment.
