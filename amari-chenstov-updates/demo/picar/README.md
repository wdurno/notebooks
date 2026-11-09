# PiCar Phase 0D

Self-contained software runtime: run commands from this repository's root.
Nothing imports another checkout, downloads a model, connects to a remote
robot, or imports hardware drivers. The shipped service supports a fake driver
only. Simulation parameters and SI mappings are **unvalidated engineering
defaults**, not physical PiCar measurements.

## Installation

Python 3.11+ is supported. The Pi service/client and shared wire contract use
only Python's standard library. Copy this repository to the Pi, not a pointer
to a development checkout. There is no Torch, Qwen, Bullet or Gymnasium
requirement on the Pi. The simulator additionally requires `numpy` and
`pybullet==3.2.7` in the project's environment. The coordinator's pinned
experiment requirements live in `numerical_experiment/picar/requirements.txt`.
No install or import starts this service or triggers motion.

For fake service development, set `PICAR_SERVICE_TOKEN` to a randomly generated
secret of at least 32 bytes in the process environment, then run:

```sh
python -m demo.picar.service --fake-driver --port 8765
```

This binds **127.0.0.1 only**. All endpoints require `Authorization: Bearer
<token>`. For future deployment use an operator-created authenticated SSH
tunnel, never a public HTTP port. No physical adapter or calibration routine
is provided. Adding one requires a separate explicit hardware authorization
and measured servo/speed calibration. Driver `apply` and `stop` must be bounded,
nonblocking local operations; a software watchdog cannot guarantee a stop
against a wedged hardware driver or power electronics. A physical emergency
stop must remain independent.

## Shared Contract

`src.robotics` exports frozen `Control`, `RobotCommand`, `RobotObservation`,
`CommandReceipt`, `RewardComponents`, `StepResult`, and `RobotBackend`.
`Calibration` owns only robot settings, not the experiment configuration.

- Control order is pan, tilt, steering, drive. Pan is left-positive, tilt
  up-positive, steering right-positive. Angles are radians; drive is m/s.
- Camera angles are **absolute** relative to the chassis. Nominal tilt is
  -8 degrees, not electrical servo zero. Stop holds camera angles while
  setting steering and drive to zero. Steering alone never enables drive.
- Probability mixtures preserve Plan19's exact eight-vector table, then
  calibrate once. Forward/reverse limits are .20/.10 m/s; steering is bounded
  to 25 degrees; pan to 60 degrees; tilt ranges from -8 to +30 degrees.
- `executed_controls` means the accepted actuator setpoint, not measured
  velocity or privileged pose. Simulator steering gain acts in the plant;
  receipts do not expose its latent value.
- Command timestamps use the backend's clock, **not the learner machine's**
  wall clock. Use the timestamp on an observation to issue a short-lived
  command; expiry covers transport delay. Service timestamps are monotonic;
  simulator timestamps are simulation seconds. Wall time is recorded apart.
- A receipt acknowledges application, not guaranteed motion completion. An
  exact retry returns the original receipt with `duplicate=True` and does not
  reapply or extend motion. Sequence conflicts and old IDs are rejected, even
  after the bounded receipt cache evicts the old response.

`src.robotics.client.RobotClient` constructs without network activity. Call
`connect(run_id=...)`, `observe()`, and explicitly `arm()` before `command()`.
Reconnect creates a fresh disarmed session and clears stale command identity.
No automatic arm, motion retry, or redirect following occurs.

Authenticated routes: `GET /health`, `GET /observe`; `POST /session`, `/arm`,
`/stop`, `/command`. State changes reject GET. Payloads are bounded to 8 KiB,
connection reads to two seconds; the client also bounds replies. Commands end
at the earliest of duration, expiry and watchdog. Re-arm retries cannot extend
an existing deadline. Watchdog, stop and camera/driver faults disarm. A latched
actuator fault requires a local restart. Camera I/O releases the control lock
so a stalled capture cannot starve stopping. The server owns its watchdog
thread, independent of HTTP request completion or inference.

## Simulator API

```python
from src.robotics import Control
from src.robotics.simulator import PyBulletBackend, SimulatorConfig

cfg = SimulatorConfig(realtime=False, width=160, height=120)
with PyBulletBackend(cfg, run_id="smoke") as sim:
    observation = sim.reset(seed=190001)
    result = sim.step(Control(drive_mps=.20))
    checkpoint = sim.snapshot()  # JSON-compatible; owner persists atomically
```

`step(Control | RobotCommand, training=True)` performs one nominal interval
and returns `StepResult`. The result contains no shaped reward or return:
`reward_components` contains actual elapsed seconds, nominal interval units
for step cost, terminal bonus and terminal-potential-zero flag. Combine these
with cached RGB-only potentials using `src.continual.picar_math`'s numerical
contract. Deadline is a true finite-horizon termination, not truncation;
infrastructure interruption remains incomplete and must resume or be reported
as such. Terminal components are emitted once. Worker captures aggregate
elapsed time since the previous result, even across `capture=False` ticks.

For independent production workers use `command(RobotCommand)` and
`advance(physics_steps, training=True, capture=True)`. `advance` **never
sleeps**, so the caller must pace it at 240 Hz and own bounded queues/deadline
decisions; no GPU scheduler is implemented here. Physics must continue when
inference misses a deadline. Expired setpoints become a stopped fallback.
`step` defaults to paced operation; `realtime=False` is labeled accelerated
diagnostics. Do not use synchronous `step` as a production inference scheduler.

`observe()` and `render()` expose only on-robot RGB/timing/accepted controls.
`render()` returns HWC uint8 NumPy RGB; observations use packed RGB bytes.
`render_topdown()` and `diagnostics()` are **privileged administrative APIs**,
not learner input. The latter contains pose, gain, drift, layout and terminal
geometry. Horizontal FOV is 90 degrees; vertical FOV is derived from aspect.

Reset jitter and mirrored layouts are seeded with a local NumPy generator.
A fixed conservative passage/visibility screen runs on reset. This checks
geometry, not successful learned navigation; a scripted feasibility pilot is
still required before scientific acquisition. Vehicle motion comes solely
from Bullet rigid-body, wheel-contact and revolute-joint dynamics. Procedural
checker-floor intensities and steering gain interpolate at physics ticks on
the Plan19 exposure clock. Evaluation (`training=False`), reset and sleep do
not advance exposure. Reset preserves exposure unless explicitly overridden.

Terminal clearance is measured from the front bumper **center** (.16 m ahead
of the chassis origin) to the target sphere's horizontal circular surface
(radius .10 m). Heading is chassis-forward toward the target center. Success
needs [.25,.45] m clearance, <=15 degrees heading error, <=.02 m/s speed and
one continuous second without object/wall/target contact. Floor contact is
excluded. Collision wins over success; final-tick success wins over deadline.

## Durable Resume And Tests

`snapshot()` returns a detached, SHA256-checked dictionary containing resolved
robot configuration, seed, session/sequence/observation metadata, capture
accounting and this episode's command/physics events. `restore(snapshot)`
requires matching configuration/run; `PyBulletBackend.from_snapshot(snapshot)`
constructs a new backend. Reconstruction replays physics from seeded reset,
preserving solver/contact history rather than approximating from pose alone.
It costs O(episode physics ticks); no process-local `saveState` handle is a
checkpoint. Schema, Bullet package/API and checksum mismatches fail clearly.
Exact reproducibility is tested on the pinned CPU stack, not promised across
architectures or different numerical libraries. Persistence/fsync/immutable
publication belongs to the experiment's storage owner.

`restore(snapshot)` preserves capture/sequence counters and returns `None`.
Restore the saved runner observation alongside it. `capture=True` explicitly
requests a new observation instead; this changes frame identity.

## Independent Realtime Worker

`src.robotics.worker.SimulatorWorker(config, run_id=...)` uses a spawned child
process and bounded local request/reply queues. Use it as a context manager,
then `reset(seed=..., training=True)`. `command(RobotCommand)` submits the
same bounded protocol. `observe()` captures policy-visible RGB; `harvest()`
returns elapsed/reward components accumulated since the previous harvest.
Neither call is required for physics to continue while the parent learns.
Periodic captures run in the child. `diagnostics()`/`render_topdown()` remain
privileged. Construction alone starts no process; explicit `start()` or
context entry starts it. No hardware or remote robot is involved.

Only realtime configs are accepted. Each actual tick sets its next deadline;
missed ticks are counted and discarded, never accelerated as catch-up.
`worker_missed_physics_ticks` and wall/simulator clocks reveal stalls instead
of concealing them. IPC queues have bounded item count and 64 MB payloads;
the child never waits on a full reply queue. RPC timeout is ambiguous for a
motion acknowledgment, so reconcile using sequence deduplication; no motion
is automatically retried. `stop()` zeros controls while the task clock keeps
running. `pause()` stops both controls and active-task time for explicit sleep
or administration, never to make inference appear to meet deadlines.

`snapshot()` wraps the backend snapshot with active/training flags;
`restore(snapshot)` preserves counters and does not invent a capture. It also
accepts a raw backend snapshot. Snapshot publication is a serial physics-tick
boundary, not a claim that parent memory and physics state are automatically
one transaction. The runner checkpoints its pending actions, last observation,
learner/RNG state and this snapshot together. `resume(training=...)` exits an
administrative pause. The owner closes on parent exit; context exit joins and,
if necessary, terminates only this simulator process.

Spawned workers require an importable entry point with the normal
`if __name__ == "__main__":` guard. `harvest()` is an accumulated endpoint,
not a durable per-interval transition stream: production acquisition still
must record interval captures, accepted receipts and fallback controls, and
must not count one delayed harvest as one nominal acquisition. Restoring an
active snapshot resumes physics immediately; use an explicitly paused safe
boundary when startup must restore other components before time resumes.

```sh
python -m pytest -q test/unit/test_picar_robot*.py
python -m pytest -q test/integration/test_picar_simulator*.py
```

These tests are bounded, deterministic CPU checks, including a fresh-process
JSON restore/continuation and an ephemeral loopback fake-service test. A
restricted sandbox may require local-socket permission for that one test.
No model downloads, remote connections, or real hardware actions occur.
