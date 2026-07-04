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


Replay a fitted policy against cached Phase 1 encodings before running the robot:

```bash
PYTHONPATH=src ~/.venv/bin/python scripts/run_phase2_replay.py \
  --fit-id <RUN_ID> \
  --max-windows 4 \
  --device auto
```

Omit `--fit-id` to replay the latest fit containing `policy.pt`.
Use `--checkpoint-path` to load a checkpoint directly.

Run the live Phase 2 robot loop:

```bash
PYTHONPATH=src ~/.venv/bin/python scripts/run_phase2_robot.py \
  --picar-host 10.0.0.224:5000 \
  --data-root artifacts/data/phase2-live \
  --fit-id <RUN_ID> \
  --x-resize 160 \
  --y-resize 120
```

Omit `--fit-id` to run the latest fit.
The live loop runs until `Ctrl-C` unless `--steps` is provided.
Use `--steps 5` for a short startup smoke test.
Use at least `context_steps + prediction_steps` steps to exercise the LSTM path; for the first window fit, use `--steps 12`.
Speech input, speech output, and reward scoring are enabled by default.
Pass `--no-speech-input`, `--no-speech-output`, or `--no-reward` to disable them.

View metrics in:

```text
experiments/reports/phase2_fit_metrics.ipynb
```

Leave `FIT_ID` empty in the notebook to inspect the newest fit.
Set `FIT_ID` to a run id when comparing a specific fit.
