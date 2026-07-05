# Setup

Use the GPU server for AI work.
Use the Raspberry Pi only as the light robot server.

Server setup:

```bash
~/.venv/bin/python -m pip install -r requirements-server.txt
```

Optional Hugging Face authentication:

```bash
huggingface-cli login
```

A token is not required, but authenticated downloads are faster and less rate limited.

Robot setup:

```bash
pip install -r requirements-robot.txt
```

Keep server dependencies off the Raspberry Pi.
`torch`, `transformers`, Qwen weights, STT, and TTS belong on the GPU server.

Useful checks:

```bash
~/.venv/bin/python -m pytest -q
PYTHONPATH=src ~/.venv/bin/python scripts/check_speech_io.py --playback
```

Use `robot_install.md` for Raspberry Pi wheel install and Flask server startup.
