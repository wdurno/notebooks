# Experimental Interface

This package implements requirement 5 from `demo/AGENTS.md`.

Primary entrypoint:

- `src/experiments/experiment_interface.py`

Run from the `demo/` directory:

```bash
python -m src.experiments.experiment_interface --help
```

## What it does

- Creates a new UUID for each run and prints it at startup.
- Persists observations in `data/<uuid>/observations.jsonl`.
- Persists compressed frame blobs in `data/<uuid>/images/step_*.npz`.
- Persists snapshots in `model/<uuid>/snapshots/`.
- Stores only tunable parameters in snapshots, plus SSR sufficient statistics and replay metadata.
- Keeps at most `--snapshot-keep` snapshots (default `3`).
- Runs until interrupted with `Ctrl-C`.

## Core CLI flags

- `--phase {init,tune,retask}`
- `--fixed-t FLOAT` (optional)
- `--t-step FLOAT` (default `0.001`)
- `--t-log-every INT` (default `1`)
- `--load-snapshot PATH` (optional)
- `--reward-prompt TEXT` (default `"find the red ball"`)
- `--snapshot-keep INT` (default `3`)
- `--data-root PATH` (optional, default `demo/data`)
- `--model-root PATH` (optional, default `demo/model`)

## Experimental phases

### 1) Initialize data (`t=0`, no tuning)

```bash
python -m src.experiments.experiment_interface \
  --phase init \
  --fixed-t 0.0
```

Behavior:

- No training/memorization loop is triggered.
- One initial snapshot is written to `model/<uuid>/snapshots/`.

### 2) Policy tuning (`t` traverses 0 -> 1)

```bash
python -m src.experiments.experiment_interface \
  --phase tune \
  --t-step 0.001 \
  --t-log-every 1
```

Behavior:

- `t` traverses linearly in steps with visible logs.
- Snapshot is written after each memorization event.

### 3) Retasking (`t=1` by default, tuning enabled)

```bash
python -m src.experiments.experiment_interface \
  --phase retask \
  --reward-prompt "find the blue cube" \
  --load-snapshot model/<prior-uuid>/snapshots
```

Behavior:

- New UUID directories are always created.
- Prior snapshot loading is optional.
- If `--fixed-t` is not passed, retask defaults to `t=1`.

## Output layout

```text
data/<uuid>/
  observations.jsonl
  run_meta.json
  images/
    step_000000.npz
    step_000001.npz

model/<uuid>/
  run_meta.json
  snapshots/
    snapshot-step-000000-initial.pt
    snapshot-step-000640-memorize-mem-0064.pt
```
