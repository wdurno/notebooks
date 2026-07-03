# Run Phase 2

Phase 2 fits and evaluates the KL-projection LSTM policy from Phase 1 data.
Run long jobs in `tmux`.
Use a UTC timestamped `RUN_ID` so fit artifacts and summaries line up cleanly.
Training uses `--device auto` by default, which selects CUDA when available and CPU otherwise.
Pass `--device cpu` for an explicit CPU run.

Install server dependencies:

```bash
pip install -r requirements-server.txt
```

Precompute visual encodings before fitting:

```bash
PYTHONPATH=src ~/.venv/bin/python scripts/precompute_phase2_encodings.py \
  --data-root artifacts/data/phase1
```

If the VLM is not already local, allow the model download explicitly:

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

Model artifacts are written to:

```text
artifacts/models/phase2/<RUN_ID>/
```

Compact summaries are written to:

```text
experiments/runs/phase2/<RUN_ID>/summary.json
```

View metrics in:

```text
experiments/reports/phase2_fit_metrics.ipynb
```

Leave `FIT_ID` empty in the notebook to inspect the newest fit.
Set `FIT_ID` to a run id when comparing a specific fit.
