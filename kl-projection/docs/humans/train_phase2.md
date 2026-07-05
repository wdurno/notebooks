# Train Phase 2

Phase 2 fits the KL-projection LSTM from Phase 1 observations.
Run long jobs in `tmux`.

Precompute visual encodings:

```bash
PYTHONPATH=src ~/.venv/bin/python scripts/precompute_phase2_encodings.py \
  --data-root artifacts/data/phase1
```

Allow model downloads only when needed:

```bash
PYTHONPATH=src ~/.venv/bin/python scripts/precompute_phase2_encodings.py \
  --data-root artifacts/data/phase1 \
  --allow-model-downloads
```

Start a window-sampling fit:

```bash
mkdir -p artifacts/models/phase2
RUN_ID="phase2-window-$(date -u +%Y%m%dT%H%M%SZ)"

PYTHONPATH=src ~/.venv/bin/python scripts/train_phase2_kl.py \
  --run-name "$RUN_ID" \
  --fit-mode window_sampling_fit \
  --validation-fraction 0.2 \
  --context-steps 8 \
  --prediction-steps 4 \
  --window-stride 1 \
  --batch-size 4 \
  --epochs 200 \
  --learning-rate 1e-3 \
  --device auto \
  2>&1 | tee "artifacts/models/phase2/${RUN_ID}.train.log"
```

Artifacts:

```text
artifacts/models/phase2/<RUN_ID>/
experiments/runs/phase2/<RUN_ID>/summary.json
```

Replay before a robot trial:

```bash
PYTHONPATH=src ~/.venv/bin/python scripts/run_phase2_replay.py \
  --fit-id <RUN_ID> \
  --max-windows 4 \
  --device auto
```

Omit `--fit-id` to use the newest fit.
Use `--checkpoint-path` to load a checkpoint directly.

View metrics:

```text
experiments/reports/phase2_fit_metrics.ipynb
```

Leave `FIT_ID` empty in the notebook to inspect the newest fit.
