# PiCar Runtime And Optional Sidecars

Self-contained software runtime: run commands from this repository's root.
Nothing imports another checkout. Imports and construction of the fake service
and speech wrapper do not download assets or open devices. Fake operation is
the default; `hardware.py` adds an explicit physical launch and `speech.py`
adds server-side recorded-audio transcription and a causal event ledger.
Simulation parameters and SI mappings are **unvalidated engineering defaults**,
not physical PiCar measurements. No physical validation is claimed.

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
<token>`. For deployment use an operator-created authenticated SSH
tunnel, never a public HTTP port. Physical launch requires separate explicit
hardware authorization and measured servo/speed calibration as described below.
Driver `apply` and `stop` must be bounded,
nonblocking local operations; a software watchdog cannot guarantee a stop
against a wedged hardware driver or power electronics. A physical emergency
stop must remain independent.

## Optional Physical Driver

`PiCarDriver` implements the existing `Driver` API with injected SunFounder
rear wheels, front wheels, pan/tilt servos and an OpenCV camera. Only
`open_physical_driver()` imports `picar`/`cv2`, calls setup, opens the camera
and initializes actuators. Importing either module, asking for CLI help or
starting a fake service cannot do this. No sibling project is required.

For a later authorized PiCar-V operator session:

1. With motor power isolated, install the Pi's I2C support (`i2c-tools` and
   `python3-smbus`), enable I2C and grant the service account local camera/I2C
   device access. Use a Pi-compatible OS and Python environment exposing the
   OS SMBus package, for example a venv with `--system-site-packages`.
2. Review the external SunFounder source/license, then install
   `demo/picar/requirements-robot.txt` on the Pi. Its `SunFounder_PiCar`
   distribution name and exact commit
   `9890ab7d11dbce10c4bb6f0bffe61b7b08783c5e` were verified against primary
   upstream metadata and Git refs. GPIO/OS compatibility and installation
   have not been tested here. Do not install server speech,
   Torch or Qwen on the Pi. See `PROVENANCE.md` before redistribution.
   Installation command: `python -m pip install -r demo/picar/requirements-robot.txt`.
3. Measure each servo's electrical zero, signed degrees per SI radian,
   mechanical travel and link direction. SI pan is left-positive, tilt
   up-positive, steering right-positive. Vendor offsets are reset to zero;
   all offsets belong in this calibration, not personal vendor config files.
   Wheel constructors use an empty config source (`/dev/null`), and the
   underlying steering servo offset is cleared. The wrapper's default
   +/-45 degree electrical clamp is replaced by the measured travel limits.
   Vendor constructors initially command 90 electrical degrees; the physical
   factory rejects calibration whose measured safe travel excludes that pose.
   Tilt zero means chassis-level viewing, not the nominal -8 degree pose.
4. In a separate controlled, authorized calibration session, measure forward
   and reverse PWM per m/s, motor polarity and a conservative PWM limit.
   This adapter assumes a linear open-loop speed mapping; validate the
   usable range under the actual load or reduce it. It does not measure
   achieved speed, compensate dead zones or establish physical safety.
5. Store a local JSON calibration under ignored `cache/picar/`. Required
   top-level fields are `pan`, `tilt`, `steering`, `forward_pwm_per_mps`,
   `reverse_pwm_per_mps`, `pwm_limit` (1--100), and `drive_sign` (+1/-1).
   Each servo object has `zero_deg`, signed `degrees_per_rad`, `min_deg`,
   `max_deg`. Optional `control_limits` contains the shared `Calibration`
   fields. Those defaults remain unvalidated; explicitly reduce them to the
   measured safe envelope. `HardwareCalibration.from_dict()` rejects ranges
   that would saturate servo/PWM travel, so receipts never silently report
   an unreachable SI setpoint. No guessed physical calibration is shipped.
6. Verify signs and stop with drive power isolated before any powered trial.
   Keep an independent emergency stop available. Explicit physical launch
   initializes servos and stops motors, but never arms drive automatically.

```sh
# Fake is still the default, including this new entry point:
python -m demo.picar.hardware --port 8765
# Later authorized physical deployment only, with PICAR_SERVICE_TOKEN set:
python -m demo.picar.hardware --physical \
  --calibration cache/picar/local-hardware-calibration.json \
  --camera-device 0 --width 160 --height 120 --port 8765
```

`create_service(token=..., physical=True, calibration=...)` is the equivalent
factory. It returns the existing service API with the same authentication,
session/sequence deduplication, bounds, expiry, watchdog and explicit arming.
`driver_factory` is an injection seam for tests. Closing this factory-created
physical service stops actuators and releases the camera, including bind or
startup failures in the CLI. The original `service --fake-driver` is unchanged.

The driver converts BGR uint8 to packed RGB without resizing or overlays and
rejects failed reads, unexpected dimensions/dtype and unavailable resolution.
Service observations retain robot monotonic capture-start and separate wall
timestamps; the driver does not replace them with inference/server time.
Actuator, bounds and camera faults stop and latch this driver until restart.
Stop holds camera pose and independently attempts motor stop, zero PWM and
neutral steering. Motor polarity reversals stop at zero PWM first. Electrical
servo outputs round to whole degrees for the vendor API; PWM is integer.
Both have final defensive saturation; protocol SI violations
are rejected, not silently clipped. A blocked camera read cannot hold the
actuator lock. Camera backend reads/releases and vendor actuator calls may
still hang; a software watchdog is not a hardware emergency stop.

## Server Instruction And Speech API

`speech.py` uses the standard library for typed/pretranscribed input and the
ledger. Optional real inference runs on the server, not the Pi. Install its
declared optional dependencies separately:

```sh
python -m pip install -r demo/picar/requirements-speech.txt
```

`FasterWhisperTranscriber(STTConfig(...))` loads only on the first worker
transcription. The default pinned model downloads into this checkout's ignored
`cache/picar/speech/hub/`. `STTConfig(model_path=Path(...))` reuses a local model
without Hub access; `allow_downloads=False` requires cached assets. No sibling
assets are required. The default model's commit comes from the prior design's
locally recorded Hub ref; no new download or inference validation was performed.
Asset/license checks remain part of deployment. CPU/int8 and two inference
threads are default; GPU use must be selected explicitly and budgeted with
learning. Construction does not load a model or start a worker.

```python
from pathlib import Path
from demo.picar.speech import (
    AudioUtterance, FasterWhisperTranscriber, InstructionLedger, SpeechBridge,
)

# All times below use the SAME absolute clock as decision_time, not wall time.
history = InstructionLedger(clock_domain="episode_seconds")
history.typed("operator-1", "Approach the red ball", at=1.)
history.transcript("recorded-1", "No, the blue ball", utterance_start=2.,
                   utterance_end=3., available_at=5., kind="correction",
                   target_id="operator-1")
history.typed("done-1", "Task complete", at=8., kind="completed", target_id="recorded-1")
contexts = history.context(as_of=6.)
statuses = history.model_status_events(as_of=6.)  # batch["status_events"]
restored = InstructionLedger.from_snapshot(history.snapshot())

# Real optional STT, for a finalized recorded file or an upstream audio producer:
# bridge = SpeechBridge(FasterWhisperTranscriber(), clock=coordinator_clock).start()
# bridge.submit(AudioUtterance("audio-2", Path("cache/picar/utterance.wav"), 9., 10.),
#               ledger=history)
# outcomes = bridge.drain(history)  # poll later, never wait in a control tick
# bridge.close()                   # inspect bool result and abandoned_ids
```

Coordinator contract: `context(as_of)` exports **only** `text`, `issued_at`,
`completed_at`, compatible with `models.LanguageEvent`. `issued_at` is text
availability, never utterance start. Future events and future completion times
are absent. Corrections close their target and add the replacement; interruption
closes the target without inventing a new motor action. Completion, correction,
interruption and reply/status text also export via `status_events(as_of)` as
`{text, available_at, kind}` with their original ledger kinds. For the model
batch use `model_status_events(as_of)`: it preserves text and availability
but maps every kind to `robot_status`, matching the model's status whitelist.
Neither export mutates the ledger. Original IDs, source, utterance start/end,
availability and target IDs remain in `events(as_of)`, `get(id)` and the snapshot.
Stable event IDs are the audio producer's utterance IDs; revisions use new IDs
and `target_id`, not mutations. Identical deliveries deduplicate; conflicting
IDs fail. Teacher, learner and replay should consume these same exports at each
decision's recorded time and persist the ledger with the trajectory. This
sidecar does not modify their model, runner or learner wiring.
The ledger is coordinator-owned: serialize frontend admissions and snapshots
there; the background worker only produces outcomes, never mutates history.

`SpeechBridge` follows the earlier stream design's separation of ingestion
and inference. A caller supplies finalized recorded audio or transcripts; no
microphone or speaker is opened. `submit(utterance, ledger=history)` checks
restored IDs and enqueues immediately; capacity includes in-flight and
completed-but-undrained results. Full queues raise `queue.Full`, never silently
drop instructions. `drain(history)` admits outcomes without waiting for STT;
text becomes available only after the lazy segment iterator finishes.
`TranscriptionOutcome` records queue/start/finish times, inference seconds and
explicit errors. Include these costs and queue delays in later operational
runs. Failed or empty transcripts are not instructions; retry is explicit.

Run this worker outside capture/control/watchdog callbacks. The Pi watchdog
remains in its separate service process. The server worker is a daemon thread,
not hard realtime isolation; native inference cannot be forcibly cancelled.
`close(timeout=...)` cancels queued admission, lists `abandoned_ids`, and returns
false if inference still runs. Persist the ledger and finalized audio IDs at a
drained boundary; after interruption resubmit unfinished IDs with the restored
ledger. Worker queues are not durable checkpoints. Never reinterpret timestamps
from a different machine or a restarted monotonic clock without an explicit
coordinator clock mapping. This adds neither TTS nor per-action generation;
ordinary student motor actions remain the LSTM's responsibility.

Focused no-hardware acceptance (no sockets, models, microphone or network):

```sh
~/.venv/bin/python -m pytest -q test/unit/test_picar_hardware.py test/unit/test_picar_speech.py
```

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
