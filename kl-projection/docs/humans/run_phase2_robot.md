# Run Phase 2 Robot

Start the Raspberry Pi server first.
See `robot_install.md`.

Run live Phase 2:

```bash
PYTHONPATH=src ~/.venv/bin/python scripts/run_phase2_robot.py \
  --picar-host 10.0.0.224:5000 \
  --data-root artifacts/data/phase2-live \
  --fit-id <RUN_ID> \
  --x-resize 160 \
  --y-resize 120
```

Omit `--fit-id` to use the newest fit.
The loop runs until `Ctrl-C` unless `--steps` is provided.
Use at least `context_steps + prediction_steps` steps to exercise the LSTM path.
For the first window fit, use `--steps 12` as a smoke test.

Speech input, speech output, and reward scoring are enabled by default.
Disable them explicitly when needed:

```bash
PYTHONPATH=src ~/.venv/bin/python scripts/run_phase2_robot.py \
  --picar-host 10.0.0.224:5000 \
  --no-speech-input \
  --no-speech-output \
  --no-reward \
  --steps 12
```

Validate live wiring without hardware:

```bash
PYTHONPATH=src ~/.venv/bin/python -m pytest -q \
  tests/integration/phase2/test_phase2_live_fake_runtime.py \
  -s
```

Validate a saved robot run:

```bash
PICAR_PHASE2_RUN_DIR=artifacts/data/phase2-live/<UUID> \
PYTHONPATH=src ~/.venv/bin/python -m pytest -q \
  tests/integration/phase2/test_phase2_live_run_artifact.py \
  -s
```

The saved-run validator is intentionally marked as UX debt.
It should become a normal script with `--run-dir` and `--latest` options.
