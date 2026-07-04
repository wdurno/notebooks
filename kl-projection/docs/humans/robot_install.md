# Robot Install

The Raspberry Pi uses a light wheel install.
The GPU server does not need this path during normal development.

Build the wheel on the server machine:

```bash
~/.venv/bin/python scripts/build_robot_wheel.py
```

Copy the wheel to the Raspberry Pi:

```bash
scp dist/picar_kl-0.1.0-py3-none-any.whl pi@<robot-host>:/home/pi/picar_kl-0.1.0-py3-none-any.whl
```

Install robot dependencies on a fresh Raspberry Pi checkout:

```bash
pip install -r requirements-robot.txt
```

Reinstall just the wheel after code changes:

```bash
pip install --force-reinstall --no-deps '/home/pi/picar_kl-0.1.0-py3-none-any.whl'
```

Run the robot Flask server:

```bash
picar-kl-robot-server --host 0.0.0.0 --port 5000
```

The server auto-detects the first readable camera index.
For debugging, pass an explicit index like `--camera-index 1`.

For a no-camera hardware check:

```bash
picar-kl-robot-server --host 0.0.0.0 --port 5000 --no-camera
```

For a no-hardware smoke check:

```bash
picar-kl-robot-server --dry-run
```

Keep server/training dependencies off the Raspberry Pi.
Use `pip install -r requirements-server.txt` on the GPU server.
