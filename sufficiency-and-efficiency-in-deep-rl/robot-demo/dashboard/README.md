# PiCar Dashboard

A lightweight Python web dashboard for manually testing your PiCar V through the installed `car_env` client package.

It provides:
- Clickable drive controls: forward, backward, left, right, stop.
- Clickable camera controls: up, left, right, forward(center).
- Keyboard controls: `W/A/S/D` to drive, arrow keys for camera (`ArrowDown` = camera forward/center).
- A camera view that refreshes at **most 2 times per second**.

This app is wired to the real interface from `picar-v-rl-env`:
- `car_env.car_client.drive_left/right/forward/backward(host)`
- `car_env.car_client.look_left/right/up/forward(host)`
- `car_env.car_client.img(host, x_resize=?, y_resize=?)`

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
- In upstream `picar-v-rl-env`, movement endpoints are timed and there is no `stop` function in `car_client.py`. The dashboard tries `http://<host>/stop` as a fallback for custom APIs.
- `car_env.car_client.img(...)` returns BGR-like channel ordering; this dashboard swaps red/blue before display so colors appear correct.
- If camera updates feel unstable, set `DASHBOARD_CAMERA_HZ=1` to reduce API load.

## Example

```bash
export CAR_ENV_HOST="192.168.1.42:5000"
export CAR_ENV_X_RESIZE="80"
export CAR_ENV_Y_RESIZE="60"
python3 app.py
```
