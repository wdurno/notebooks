# Robot Legacy Import

Robot API code copied from:

`../../picar-v-rl-env/`

The copied Flask server is staged under `legacy_car_env`.
It is intended for Raspberry Pi use and may import PiCar hardware libraries at module import time.
Keep hardware imports isolated from ordinary server-side tests and GPU-server code.

