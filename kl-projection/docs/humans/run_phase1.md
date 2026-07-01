# Run Phase 1

Phase 1 generates VLM-only robot data.
The server runs the VLM, listens for operator speech, speaks robot responses, and sends action vectors to the Raspberry Pi Flask server.
By default, Phase 1 runs until `Ctrl-C`.

Install server dependencies:

```bash
pip install -r requirements-server.txt
```

Run with Qwen2.5-VL from an installed package:

```bash
picar-kl-phase1 \
  --picar-host <robot-host:5000> \
  --data-root artifacts/data/phase1
```

Run from a checkout without installing the package:

```bash
PYTHONPATH=src ~/.venv/bin/python -m picar_kl.phase1.cli \
  --picar-host <robot-host:5000> \
  --data-root artifacts/data/phase1
```

Use `Ctrl-C` to stop an open-ended collection run.
Completed steps are written as the robot runs.
The run metadata records whether the run completed or was interrupted.

Limit the run when you want a bounded smoke test:

```bash
picar-kl-phase1 \
  --picar-host <robot-host:5000> \
  --data-root artifacts/data/phase1 \
  --steps 5
```

Use a fixed action for a quiet no-VLM smoke run:

```bash
picar-kl-phase1 \
  --picar-host <robot-host:5000> \
  --data-root artifacts/data/phase1 \
  --steps 1 \
  --smoke-fixed-action look-forward \
  --no-speech-input \
  --no-speech-output
```

Speech input runs as a background listener by default.
Completed utterances are drained into the next Phase 1 step without requiring Enter per action.
TTS output is enabled by default.
Disable either side explicitly when needed:

```bash
picar-kl-phase1 \
  --picar-host <robot-host:5000> \
  --no-speech-input \
  --no-speech-output
```

Adjust speech detection if the room is noisy or too quiet:

```bash
picar-kl-phase1 \
  --picar-host <robot-host:5000> \
  --speech-amplitude-threshold 0.015 \
  --speech-silence-seconds 0.8 \
  --speech-min-seconds 0.25
```

Generated runs are stored as JSONL records plus compressed image blobs.
Raw latency events are recorded per step.
