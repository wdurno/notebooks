# Experiment Data Layout

This directory stores ephemeral data generated while running experiments.
It is intended for outputs that may be large, numerous, or short-lived and should
not be committed to git.

Examples include:

- rollout dumps
- replay snapshots
- camera captures
- training logs
- evaluation traces
- intermediate tensors or feature blobs
- exported metrics and summaries

## Design goals

- Keep generated data out of source control.
- Give each experiment its own directory.
- Make it easy to write many output blobs without naming collisions.
- Keep lightweight metadata near the data it describes.
- Support partial automation and manual robotics workflows.

## Directory structure

```text
data/
  <experiment-name>/
    metadata.json
    blobs/
    logs/
    metrics/
    artifacts/
```

The top-level unit of organization is one directory per experiment run or run
family. Code should create a new experiment directory before writing outputs.

## Recommended naming

Use stable, sortable names so runs are easy to inspect from the shell.

Examples:

- `2026-02-28-policy-debug-v1`
- `2026-02-28-qwen-finetune-exp-001`
- `2026-02-28-picar-camera-eval-night`

If one experiment is resumed multiple times, either:

- keep all outputs in the same experiment directory and append new blobs, or
- create a new directory with a clear suffix such as `-resume-01`

Choose one convention and keep it consistent.

## Subdirectories

### `metadata.json`

Small, human-readable run metadata.

This file should describe the run rather than duplicate the large data files. It
is a good place for:

- experiment name
- start time
- code revision or git commit
- model identifiers
- dataset or environment identifiers
- important hyperparameters
- robot or hardware notes
- free-form comments

### `blobs/`

Primary data dumps.

This is the default location for large serialized outputs written during the run.
Examples:

- `.pt` tensor dumps
- `.npy` arrays
- replay buffer snapshots
- image batches
- chunked binary captures

Blob files should usually be append-only or chunked by step, episode, or time.

Example:

```text
data/
  2026-02-28-policy-debug-v1/
    blobs/
      episode-000001.pt
      episode-000002.pt
      step-010000.npy
```

### `logs/`

Textual or structured logs produced by the run.

Examples:

- stdout/stderr captures
- JSONL event logs
- robotics setup notes
- hardware error traces

### `metrics/`

Derived scalar outputs and summaries.

Examples:

- per-episode rewards
- evaluation tables
- CSV exports
- summary JSON files
- plots generated from the run

### `artifacts/`

Miscellaneous outputs that do not fit cleanly into the other categories.

Examples:

- rendered videos
- debug snapshots
- notebook exports
- temporary reports

## Usage pattern

The intended workflow is:

1. Create a new experiment directory.
2. Write a small `metadata.json` file immediately.
3. Stream large outputs into `blobs/` as the run executes.
4. Write logs and metrics into their own subdirectories.
5. Treat the entire experiment directory as disposable unless explicitly archived elsewhere.

## Robotics-specific notes

For robot-facing experiments, this directory should hold the generated data, not
the manual setup instructions themselves. Keep setup steps in code comments,
documentation, or test instructions, and record only run-specific observations in
the experiment metadata or logs.

Useful metadata fields for robotics runs include:

- robot identifier
- environment description
- camera resolution
- control frequency
- operator notes
- whether the run was manual, semi-automated, or fully automated

## Git policy

This directory is expected to be ignored by git except for this README or other
small documentation files you intentionally keep under version control.

Do not rely on data in this directory as the only copy of an important result.
If an experiment matters long-term, archive it explicitly to a durable location.
