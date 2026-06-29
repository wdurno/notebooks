# Import Provenance Notes

## Purpose

This repo will copy and refactor useful assets from prior experiments.
Do not modify the referenced source repos.
Only read from them or copy assets into this repo.

## Provenance Policy

When importing non-trivial code, notebooks, tests, data schemas, or configs from another repo, record provenance near the imported asset.
Prefer a short note in one of these places:

1. A nearby README.
2. A tracked artifact manifest.
3. A module docstring when the imported code is narrow.
4. A docs note under `/docs/agents/` when the import affects architecture.

The note should usually include:

1. Source repo-relative path.
2. Import date.
3. Why the asset was imported.
4. Whether the import was copied verbatim or refactored.
5. Any important behavior intentionally preserved.

## Artifact Boundary

Raw observations, sampled experimental data, checkpoints, and large model artifacts belong in ignored artifact directories.
Distilled run outputs belong in `/experiments/runs/` only when they are small, high-value, and under the 1MB per-run limit.

The intended pattern is an abstraction hierarchy.
Data should get smaller as it moves from raw observations to derived metrics, summaries, reports, and manifests.
