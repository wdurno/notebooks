# Build

The only package build is the light robot wheel.
The GPU server usually runs from the checkout and does not need a wheel.

Build the wheel:

```bash
~/.venv/bin/python scripts/build_robot_wheel.py
```

Output goes to:

```text
dist/picar_kl-0.1.0-py3-none-any.whl
```

`build/` and `dist/` are ephemeral and ignored by git.

Install or reinstall the wheel on the Raspberry Pi:

```bash
pip install --force-reinstall --no-deps '/home/pi/picar_kl-0.1.0-py3-none-any.whl'
```

Use `robot_install.md` for the full Pi-side sequence.
