# Robot Install

The Raspberry Pi uses a light wheel install.
The GPU server does not need this path during normal development.

Build the wheel on the server machine:

```bash
~/.venv/bin/python scripts/build_robot_wheel.py
```

Install the wheel on the Raspberry Pi:

```bash
pip install 'dist/picar_kl-0.1.0-py3-none-any.whl[robot]'
```

Run the robot Flask server:

```bash
picar-kl-robot-server --host 0.0.0.0 --port 5000
```

For a no-hardware smoke check:

```bash
picar-kl-robot-server --dry-run
```

Keep server/training dependencies off the Raspberry Pi.
Use `pip install -r requirements-server.txt` on the GPU server.
