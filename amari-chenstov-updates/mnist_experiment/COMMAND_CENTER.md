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
```

The preview reuses Plan 2 replicas 6 through 10. Values shown for `selected`
and `memory-matched` replay are planning placeholders used only to estimate
cost. Plan 3 will select the replay budget at a phase gate and derive the
memory-matched capacity from measured persistent bytes. There is intentionally
no Plan 3 `prepare` or `run` command yet.
