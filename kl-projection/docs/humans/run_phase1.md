# Run Phase 1

Phase 1 generates VLM-only robot data.
The server runs the VLM and sends action vectors to the Raspberry Pi Flask server.

Install server dependencies:

```bash
pip install -r requirements-server.txt
```

Run with Qwen2.5-VL from an installed package:

```bash
picar-kl-phase1 \
  --picar-host <robot-host:5000> \
  --data-root artifacts/data/phase1 \
  --steps 10
```

Run from a checkout without installing the package:

```bash
PYTHONPATH=src ~/.venv/bin/python -m picar_kl.phase1.cli \
  --picar-host <robot-host:5000> \
  --data-root artifacts/data/phase1 \
  --steps 10
```

Use a fixed action for a no-VLM smoke run:

```bash
picar-kl-phase1 \
  --picar-host <robot-host:5000> \
  --data-root artifacts/data/phase1 \
  --steps 1 \
  --smoke-fixed-action look-forward
```

Enable speech only when audio devices and speech models are ready:

```bash
picar-kl-phase1 \
  --picar-host <robot-host:5000> \
  --enable-speech-input \
  --data-root artifacts/data/phase1 \
  --enable-speech-output
```

Generated runs are stored as JSONL records plus compressed image blobs.
Raw latency events are recorded per step.
