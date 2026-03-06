# PiCar Dashboard

A lightweight Python web dashboard for manually testing your PiCar V through the installed `car_env` client package.

It provides:
- Clickable drive controls: forward, backward, left, right, stop.
- Clickable camera controls: up, left, right, forward(center).
- Keyboard controls: `W/A/S/D` to drive, arrow keys for camera (`ArrowDown` = camera forward/center).
- A camera view that refreshes at **most 2 times per second**.
- Runtime camera resolution controls (preset or custom `x_resize` / `y_resize`) without restarting the dashboard.

This app is wired to the real interface from `picar-v-rl-env`:
- `car_env.car_client.drive_left/right/forward/backward(host)`
- `car_env.car_client.look_left/right/up/forward(host)`
- `car_env.car_client.img(host, x_resize=?, y_resize=?)`
- direct `GET /img` fallback for newer APIs that return image bytes instead of JSON

## Quick Start

1. Move into the dashboard directory:

```bash
cd dashboard
```

2. Create/activate a Python environment (optional but recommended), then install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Run the app:

```bash
export CAR_ENV_HOST="192.168.1.42:5000"
python3 app.py
```

4. Open:

```text
http://localhost:8080
```

## Configuration

Required:
- `CAR_ENV_HOST` (example: `192.168.1.42:5000`)

### Dashboard server

- `DASHBOARD_HOST` (default: `0.0.0.0`)
- `DASHBOARD_PORT` (default: `8080`)
- `DASHBOARD_CAMERA_HZ` (default: `2`, capped at 2)
- `CAR_ENV_X_RESIZE` (optional, passed to `car_env.car_client.img`)
- `CAR_ENV_Y_RESIZE` (optional, passed to `car_env.car_client.img`)

Notes:
- You can change camera resolution live from the dashboard; this maps to `car_env.car_client.img(host, x_resize=?, y_resize=?)`.
- `CAR_ENV_X_RESIZE` and `CAR_ENV_Y_RESIZE` now act as startup defaults and can be overridden from the UI.
- In upstream `picar-v-rl-env`, movement endpoints are timed and there is no `stop` function in `car_client.py`. The dashboard tries `http://<host>/stop` as a fallback for custom APIs.
- `car_env.car_client.img(...)` is used first. If that fails (for example when `/img` no longer returns JSON), the dashboard falls back to a direct `GET /img`.
- For legacy `car_client.img(...)` array payloads, the dashboard swaps red/blue before display so colors appear correct.
- The selected `x_resize`/`y_resize` is enforced by the dashboard before streaming `frame.jpg`, so resolution changes still apply even if the robot API ignores resize query params.
- On newer robot APIs, a one-time warning about `car_client.img` fallback is expected at startup; subsequent frames use direct `/img`.
- In `/home/evan/Documents/picar-v-rl-env/src/car_env/constants.py`, capture defaults to `SCREEN_WIDTH=160` and `SCREEN_HEIGHT=120`; requesting larger sizes only upscales that source frame.
- If camera updates feel unstable, set `DASHBOARD_CAMERA_HZ=1` to reduce API load.

## Example

```bash
export CAR_ENV_HOST="192.168.1.42:5000"
export CAR_ENV_X_RESIZE="80"
export CAR_ENV_Y_RESIZE="60"
python3 app.py
```
