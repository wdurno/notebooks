# Data Layout

Large artifacts stay out of git.
Small summaries may be versioned when useful.

| Path | Purpose | Git |
| --- | --- | --- |
| `artifacts/data/phase1/` | raw Phase 1 observations, images, latency | ignored |
| `artifacts/data/phase2-live/` | raw live Phase 2 robot runs | ignored |
| `artifacts/data/phase2/encodings/` | cached visual-token encodings | ignored |
| `artifacts/models/` | model weights, downloaded models, fit artifacts | ignored |
| `experiments/runs/` | compact run summaries under 1 MB | tracked when useful |
| `experiments/reports/` | analysis notebooks | tracked |
| `artifacts/manifests/tracked/` | version-safe defaults and references | tracked |
| `artifacts/manifests/ephemeral/` | machine-specific references | ignored |

Phase 1 and live Phase 2 runs store:

```text
run_meta.json
observations.jsonl
images/step_*.npz
```

`observations.jsonl` records action distributions, text context, rewards, and raw latency events.
Images are compressed NumPy arrays.

Phase 2 fitting stores model artifacts under:

```text
artifacts/models/phase2/<RUN_ID>/
```

The matching compact summary is stored under:

```text
experiments/runs/phase2/<RUN_ID>/summary.json
```
