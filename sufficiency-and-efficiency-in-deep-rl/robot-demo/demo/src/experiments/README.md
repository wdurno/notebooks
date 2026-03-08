# Experimental Interface

This package implements requirement 5 from `demo/AGENTS.md`.

Primary entrypoint:

- `src/experiments/experiment_interface.py`

Run from the `demo/` directory:

```bash
python -m src.experiments.experiment_interface --help
```

Set the PiCar API endpoint the same way as the integration tests:

```bash
PICAR_V_HOST=<host:port> python -m src.experiments.experiment_interface --help
```

## What it does

- Creates a new UUID for each run and prints it at startup.
- Persists observations in `data/<uuid>/observations.jsonl`.
- Persists compressed frame blobs in `data/<uuid>/images/step_*.npz`.
- Persists snapshots in `model/<uuid>/snapshots/`.
- Reuses the policy backbone's single 4-bit Qwen instance for reward scoring
  (reward pass runs frozen/inference-only with adapters disabled).
- Treats malformed agent JSON as invalid speech output:
  - no fallback text is spoken
  - step reward gets a default `-1.0` adjustment
- Stores only tunable parameters in snapshots, plus SSR sufficient statistics and replay metadata.
- Keeps at most `--snapshot-keep` snapshots (default `3`).
- Runs until interrupted with `Ctrl-C`.

## Core CLI flags

- `--phase {init,tune,retask}`
- `--picar-host HOST:PORT` (optional CLI override for `PICAR_V_HOST`)
- `--deterministic-coding` (disable sampling for policy text generation)
- `--history-window INT` (default `12`)
- `--all-images` (attach an image placeholder to all user messages in context)
- `--fixed-t FLOAT` (optional)
- `--t-step FLOAT` (default `0.001`)
- `--t-log-every INT` (default `1`)
- `--load-snapshot PATH` (optional)
- `--reward-prompt TEXT` (default `"find the red ball"`)
- `--snapshot-keep INT` (default `3`)
- `--data-root PATH` (optional, default `demo/data`)
- `--model-root PATH` (optional, default `demo/model`)
- `--log-level {DEBUG,INFO,WARNING,ERROR}` (default `WARNING`)

Use `--log-level INFO` for concise runtime telemetry:

- step/action summaries
- reward summaries
- robot speech (`say`)
- captured STT text

Use `--log-level DEBUG` for deep diagnostics such as shared-model adapter state,
raw generations, and prompt/token traces.

Context/image behavior notes:

- `--history-window` controls how many recent messages are retained in rolling context.
- By default, only the most recent user message is image-tagged.
- With `--all-images`, every retained user message is image-tagged (reusing the current frame for each image slot).

## Experimental phases

### 1) Initialize data (`t=0`, no tuning)

```bash
PICAR_V_HOST=<host:port> python -m src.experiments.experiment_interface \
  --phase init \
  --fixed-t 0.0
```

For deterministic control text decoding while debugging malformed JSON:

```bash
PICAR_V_HOST=<host:port> python -m src.experiments.experiment_interface \
  --phase init \
  --fixed-t 0.0 \
  --deterministic-coding
```

Behavior:

- No training/memorization loop is triggered.
- One initial snapshot is written to `model/<uuid>/snapshots/`.
- If the policy emits malformed action JSON, speech is suppressed and a `-1.0`
  reward adjustment is applied for that step.
- Model stays in inference/eval mode for rollout.

If the PiCar crashes (for example, low battery), just swap batteries and run the
same phase-1 command again. Each restart creates a new run UUID directory under
`data/`, and phase 1 is designed for aggregating many independent collection
runs.

Crash-handling workflow:

1. Start phase 1 and let it collect until failure or `Ctrl-C`.
2. Restart phase 1 with the same command after recovery.
3. Repeat until enough data is collected across multiple `data/<uuid>` runs.
4. Keep all phase-1 run directories; they are intended to be merged later for
   offline tuning + memorization.

Why phase-1 restart examples usually omit `--load-snapshot`:

- In phase 1, the model is not tuned or memorized.
- So each restart with default initialization is effectively equivalent for
  collection-only runs.
- If you intentionally started phase 1 from a custom snapshot, pass
  `--load-snapshot` again on restart to keep the same initialization source.

Example restart with explicit snapshot source:

```bash
PICAR_V_HOST=<host:port> python -m src.experiments.experiment_interface \
  --phase init \
  --fixed-t 0.0 \
  --load-snapshot model/<prior-uuid>/snapshots
```

### 2) Policy tuning (`t` traverses 0 -> 1)

```bash
PICAR_V_HOST=<host:port> python -m src.experiments.experiment_interface \
  --phase tune \
  --t-step 0.001 \
  --t-log-every 1
```

Behavior:

- `t` traverses linearly in steps with visible logs.
- Snapshot is written after each memorization event.
- Malformed action JSON is penalized by `-1.0` and does not get spoken.
- Rollout sampling runs in inference/eval mode; fit/memorize blocks switch to
  optimization/train mode and then return to eval mode.

### 3) Retasking (`t=1` by default, tuning enabled)

```bash
PICAR_V_HOST=<host:port> python -m src.experiments.experiment_interface \
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

## Phase-1 Finalization (offline tune + full memorize)

After collecting many phase-1 runs, finalize them into a fresh
`model/<new-uuid>/` state:

```bash
python -m src.experiments.phase1_finalize \
  --data-runs data/<uuid1> data/<uuid2> data/<uuid3> \
  --epochs 3 \
  --batch-size 8 \
  --fit-iters 32
```

Optional: initialize from an existing snapshot first:

```bash
python -m src.experiments.phase1_finalize \
  --data-runs data/<uuid1> data/<uuid2> data/<uuid3> \
  --load-snapshot model/<prior-uuid>/snapshots \
  --epochs 3 \
  --batch-size 8 \
  --fit-iters 32
```

What `phase1_finalize` does:

1. Loads transitions reconstructed from all provided `data/<uuid>/observations.jsonl` runs.
2. Tunes the model offline on the aggregated replay buffer.
3. Memorizes all loaded replay into SSR sufficient statistics.
4. Writes a final snapshot under a new `model/<new-uuid>/snapshots/`.

`phase1_finalize` runs in optimization/train mode for offline fit and memorize.
