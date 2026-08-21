# Phase 9 command center

See [Phase 9 experimental conditions](EXPERIMENTAL_CONDITIONS.md) for the
shared LFU mechanics and a concise definition of every condition.

Preview the five-replica controller screen without writing or running anything:

```bash
python -m mnist_experiment.command_center preview
```

Prepare its immutable configurations, then use the returned bundle path:

```bash
python -m mnist_experiment.command_center prepare
python -m mnist_experiment.command_center status --bundle <bundle-path>
python -m mnist_experiment.command_center run --bundle <bundle-path> --resume
```

`run` skips completed artifacts. `--resume` is required when an interrupted run
exists. Use `--max-runs 1` to stop after one new controller trajectory.

Add independent replicas without changing earlier bundles:

```bash
python -m mnist_experiment.command_center prepare --replicas 6-10
```

Prepare the five-replica LFU-isolation screen separately so its completed
full-LFU dependencies can be reused:

```bash
python -m mnist_experiment.command_center prepare \
  --profiles lfu-isolation --replicas 1-5
```

Profiles are selected explicitly and never form an automatic Cartesian sweep:

| Profile | Purpose |
|---|---|
| `controller-screen` | Rank-8, budget 50, axial controller search with one shared fixed control |
| `lfu-isolation` | Paired no-LFU, AC-only, and full-LFU contrasts under fixed and adaptive controllers |
| `adaptation-screen` | No-LFU controller half-lives plus a no-EWC comparison |
| `data-screen` | Opt-in sample-size and trajectory-resolution axes |
| `dense-confirm` | Opt-in dense, budget-100 confirmation |
| `smoke` | Tiny CPU integration check |

Open `results.ipynb` for completion status, paired effects, uncertainty, and
links to detailed diagnostic notebooks. Cost previews are rough planning
estimates calibrated from the Phase 8 pilot, not runtime guarantees.

## Plan 2 low-data screen

Preview the explicit three-condition screen over selected online batch sizes:

```bash
python -m mnist_experiment.plan2_command_center preview \
  --samples-per-step 1,2,4,8,16,32,64 --replicas 1-3
```

Prepare immutable prefix streams and configurations, then inspect or run the
returned bundle path:

```bash
python -m mnist_experiment.plan2_command_center prepare
python -m mnist_experiment.plan2_command_center status --bundle <bundle-path>
python -m mnist_experiment.plan2_command_center run \
  --bundle <bundle-path> --resume
```

Preparation performs no new initialization fit. Each low-`m` stream is the
first `m` ordered observations from every batch of its retained `m=128` master,
and all conditions at that `m` share the derived bundle and reference path.

Completed schema-7 runs need one inference-only pass to add 9 OvR accuracy,
9 precision, and explicit 9 recall without mutating their artifacts. Include
the reused `m=128` anchors in the same resumable pass:

```bash
python -m mnist_experiment.plan2_command_center backfill-metrics \
  --bundle <bundle-path> --device cuda --resume --include-m128-anchors
```

New schema-8 runs record those metrics directly. The results notebook averages
each primary metric across replicas separately at every environmental `p`.

The accepted Phase 3 replication uses the prepared immutable bundle
`plan2-low-data__r0004-r0005__ebf1dd35123e`. It adds replicas 4 and 5 only at
`m=8,16,32`:

```bash
python -m mnist_experiment.plan2_command_center run \
  --bundle cache/mnist_experiment/plan2/bundles/plan2-low-data__r0004-r0005__ebf1dd35123e \
  --resume
```

The package contains 18 trajectories, uses two independent retained
initializations, and records schema-8 classification metrics directly.

Phase 4 controller recalibration is complete in three accepted immutable
extensions:

- `plan2-low-data__r0001-r0005__292a833830f0`: half-life screen;
- `plan2-low-data__r0001-r0005__0c4eebbf40db`: conditional lower-bound screen;
- `plan2-low-data__r0001-r0005__69aea5f9bdc3`: explicit fixed-$\pi=.05$ audit.

The prelaunch intention `plan2-low-data__r0001-r0005__967e09afc995` was never
executed and is explicitly excluded from accepted analysis. Phase 4 selected
fixed $\pi=.05$ for confirmation; the adaptive plug-in formula remains
unresolved because its leading conditions were bound-driven.

Phase 5 independent confirmation is complete:

- source anchors: `phase9-initial__r0006-r0010__065c98b1061d`;
- paired confirmation: `plan2-low-data__r0006-r0010__7bb69d3a9424`.

The source bundle intentionally executed only its five oracle anchors via
`--oracle-anchors-only`; its other controller-screen intentions remain missing
and are not part of Phase 5. The confirmation bundle contains 20 completed
`m=8` trajectories across fresh replicas 6 through 10.

The preview uses a fixed-overhead-plus-observation cost model. Its production
anchor is the 283.68-second median from 85 completed `m=128`, `K=100`, rank-8,
budget-50 trajectories; the Phase 1 CUDA smoke was effectively flat between
`m=1` and `m=2`, so the estimate is deliberately not scaled linearly from zero.
It remains a planning proxy rather than measured GPU time.

## Plan 3 handoff

Preview the staged replay, hybrid, LFU, and deployment proposal without writing
artifacts:

```bash
python -m mnist_experiment.plan3_command_center preview
python -m mnist_experiment.plan3_command_center preview \
  --stages replay-screen --details
python -m mnist_experiment.plan3_command_center audit-handoff
```

The preview reuses Plan 2 replicas 6 through 10. The value shown for `selected`
replay is a planning placeholder used only to estimate cost. Memory-matched
replay is derived from the canonical persistent-state contract and currently
equals 25 observations. `audit-handoff` verifies the frozen bundle, run,
configuration, initialization, stream, and reference identities without
writing artifacts.

Phase 1's immutable replay-engine validation is available through:

```bash
python -m mnist_experiment.plan3_command_center prepare-phase1
python -m mnist_experiment.plan3_command_center phase1-status \
  --bundle <bundle-path>
python -m mnist_experiment.plan3_command_center run-phase1 \
  --bundle <bundle-path> --resume
```

The accepted completed bundle is
`plan3-phase1__r0006__3930db574a45`. It contains CPU/CUDA smoke runs and one
capacity-32 production timing pilot. The pure-replay preview now uses that
pilot; Fisher and LFU conditions retain the prior conservative cost model.

The paired Phase 2 replay-capacity screen is also complete. Its commands are:

```bash
python -m mnist_experiment.plan3_command_center prepare-phase2
python -m mnist_experiment.plan3_command_center phase2-status \
  --bundle <bundle-path>
python -m mnist_experiment.plan3_command_center run-phase2 \
  --bundle <bundle-path> --resume
python -m mnist_experiment.plan3_command_center analyze-phase2 \
  --bundle <bundle-path>
```

The accepted bundle is
`plan3-replay-screen__r0006-r0010__16a259169db6`; its immutable analysis is
`phase2__plan3-replay-screen__r0006-r0010__16a259169db6__adfadd6ed85f`.
Capacity 32 is the gate recommendation as the smallest replay budget that
reliably improved the principal classification outcomes over fixed-$.05$ EWC.

Phase 3's hybrid-engine validation is also complete:

```bash
python -m mnist_experiment.plan3_command_center prepare-phase3
python -m mnist_experiment.plan3_command_center phase3-status \
  --bundle <bundle-path>
python -m mnist_experiment.plan3_command_center run-phase3 \
  --bundle <bundle-path> --resume
```

The accepted smoke bundle is `plan3-phase3__r0006__a8ffdd4d6e7b`. Its eight
short runs validate capacities 0, 8, 25, and unbounded on CPU and CUDA. They
are implementation checks, not predictive comparisons.

Phase 4's paired history-mechanism frontier uses:

```bash
python -m mnist_experiment.plan3_command_center prepare-phase4
python -m mnist_experiment.plan3_command_center phase4-status \
  --bundle <bundle-path>
python -m mnist_experiment.plan3_command_center run-phase4 \
  --bundle <bundle-path> --resume
python -m mnist_experiment.plan3_command_center analyze-phase4 \
  --bundle <bundle-path>
```

The immutable bundle is
`plan3-history-frontier__r0006-r0010__12582f2d297a`. It reuses 20 completed
EWC and replay cells and adds 20 runs: replay B25 plus hybrid B8, B25, and B32
on each paired replica. Here “memory-matched” means that the B25 replay
component matches one EWC summary; the complete B25 hybrid stores both.

Phase 4 is complete. Its accepted analysis is
`phase4__plan3-history-frontier__r0006-r0010__12582f2d297a__113026398aa0`.
Hybrid B32 is the Phase 5 recommendation; hybrid B8 and B25 remain
lower-memory Pareto alternatives.

Phase 5 isolates LFU effects while reusing the accepted no-LFU controls:

```bash
python -m mnist_experiment.plan3_command_center prepare-phase5
python -m mnist_experiment.plan3_command_center phase5-status \
  --bundle <bundle-path>
python -m mnist_experiment.plan3_command_center run-phase5 \
  --bundle <bundle-path> --resume
python -m mnist_experiment.plan3_command_center analyze-phase5 \
  --bundle <bundle-path>
```

The production bundle is
`plan3-lfu-isolation__r0006-r0010__6d8c4ad67265`: 15 new trajectories and 10
immutable controls. LFU artifacts use a versioned hybrid schema and charge the
directional-ridge state to persistent memory.

The replica-6 preflight completed all three new treatment types and then
triggered the Phase 5 stop rule: directional resets were nearly universal and
PSD projection materially determined the LFU candidates. The remaining 12
runs are intentionally absent. Do not run the bundle merely to fill its grid;
see the Phase 5 preflight record in `plan3.md`.

Phase 6's oracle-free deployment frontier uses:

```bash
python -m mnist_experiment.plan3_command_center prepare-phase6
python -m mnist_experiment.plan3_command_center phase6-status \
  --bundle <bundle-path>
python -m mnist_experiment.plan3_command_center run-phase6 \
  --bundle <bundle-path> --resume
python -m mnist_experiment.plan3_command_center analyze-phase6 \
  --bundle <bundle-path>
```

The completed bundle is
`plan3-deployment-frontier__r0006-r0010__f21931201ff8`; the accepted analysis
is
`phase6__plan3-deployment-frontier__r0006-r0010__f21931201ff8__ce7440f3fe91`.
It contains 35 paired runs: current-only, fixed and adaptive EWC, fixed and
adaptive Hybrid B32, Replay B32, and unbounded replay. All EWC summaries use
EMA without LFU, and all learner costs exclude offline holdout evaluation.

Phase 7 predeclares fresh replicas 11 through 25 and executes them in blocks
of five. The initial target is ten replicas:

```bash
python -m mnist_experiment.plan3_command_center prepare-phase7 \
  --replicas 11-25
python -m mnist_experiment.plan3_command_center phase7-status \
  --bundle <bundle-path>
python -m mnist_experiment.plan3_command_center run-phase7 \
  --bundle <bundle-path> --resume --max-replicas 5
python -m mnist_experiment.plan3_command_center analyze-phase7 \
  --bundle <bundle-path>
```

Run one five-replica block at a time, analyze after each block, and honor the
recorded precision gate. Each fresh replica gets an independent $p=0$ fit and
an adaptive high-sample initial Fisher archive; no reference-optimum path is
built. The five paired treatments are current-only, fixed-$.05$ EWC, fixed
Hybrid B32, Replay B32, and unbounded replay. The artifact-only figures are in
`deployment_results.ipynb` and intentionally display only $p<.5$, although
the computational trajectory remains the unchanged 100-point path to $p=1$.

Phase 7 stopped at its ten-replica precision target. The bundle is
`plan3-fresh-confirmation__r0011-r0025__89c1451b1983`; the accepted analysis is
`phase7__plan3-fresh-confirmation__r0011-r0025__89c1451b1983__n10__85db75d1349e`.
Do not run replicas 21 through 25 merely to fill the bundle. All ten initial
Fisher estimates reached the 32,768-score maximum without meeting the 1%
relative six-sigma early-stop target; the analysis and notebook expose those
diagnostics explicitly.
