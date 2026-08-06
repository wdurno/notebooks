# Phase 9 command center

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

Profiles are selected explicitly and never form an automatic Cartesian sweep:

| Profile | Purpose |
|---|---|
| `controller-screen` | Rank-8, budget 50, axial controller search with one shared fixed control |
| `data-screen` | Opt-in sample-size and trajectory-resolution axes |
| `dense-confirm` | Opt-in dense, budget-100 confirmation |
| `smoke` | Tiny CPU integration check |

Open `results.ipynb` for completion status, paired effects, uncertainty, and
links to detailed diagnostic notebooks. Cost previews are rough planning
estimates calibrated from the Phase 8 pilot, not runtime guarantees.
