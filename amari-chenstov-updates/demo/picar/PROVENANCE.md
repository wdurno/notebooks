# Source Inspection And Licensing

Read-only reference: the sibling `kl-projection` checkout, tracked by the
enclosing repository at commit `6d6c88af8f405b44957254c4fa6c1ff7d49ed854`.
The inspected modules matched that revision; paths below are relative to the
reference checkout's root. This is provenance, not a runtime dependency or
an installation path. The checkout had no discovered LICENSE/COPYING
file or license declaration in README/pyproject metadata at inspection.
Consequently no implementation source was copied verbatim from it; no
redistribution license is asserted for its contents. The new project-local
hardware adapter is an independently written minimal functional port of the
inspected wheel/servo/camera API, not a vendored copy of `car.py`.

Inspected files and SHA256:

| Reference | Hash |
| --- | --- |
| `src/picar_kl/actions.py` | `ddea32cde3f9f6470421119f2d6cce4fe536f492ad6f8a96af2afe9f45e476cc` |
| `src/picar_kl/robot/server.py` | `adef368a837da1a3de7f7253494cb2ff032e31bf73e56b286e8dda2fa947d553` |
| `src/picar_kl/robot/legacy_car_env/car.py` | `f1fd453816510eea90ec642c2c800c9aa9fe66c9cc581a0296a0d022871d83e4` |

The eight primitive names/table are independently specified by this project's
Plan19 and preserved as declarative data with golden tests in
`src/robotics/actions.py`. The injectable service architecture was inspected;
the standard-library implementation here is newly written. The old
GET-to-move endpoints, implicit steering-triggered forward motion, import-time
hardware setup, and electrical-zero camera assumptions are not retained.

## Fix 2 Item 6 Sidecars

`demo/picar/hardware.py` ports only the vendor calls needed for rear wheel
direction/PWM/stop, front steering, pan/tilt and camera capture. It does not
import, package or depend on the reference checkout. There is no blob detector,
legacy endpoint implementation or copied import-time setup. The adapter's
calibration, shutdown and fault state are new local code. The reference hash
for `car.py` above was checked again during implementation and is unchanged.

`demo/picar/speech.py` reuses the earlier design of a lazy faster-whisper
wrapper and separate bounded ingestion/transcription queues through new local
code. It adds causal availability timestamps, stable IDs, correction/completion
events and replayable ledger exports rather than copying the old wall-clock
text-only stream. Additional read-only references inspected:

| Reference | SHA256 |
| --- | --- |
| `src/picar_kl/speech/stt.py` | `12d30796f56825b35639cfbd80dc666dba1f732fb4c1ccf4445aae971c3b2065` |
| `src/picar_kl/speech/stream.py` | `bee6da08863a2ddd9a76888b0906d39ffbbe96f2dbeb8df3b0a2c8bb3f6d6114` |
| `src/picar_kl/speech/model_store.py` | `ba7db68bdac7835585e12135cb2315b0e73488e2f26f3314bd7696d65e3c5b29` |

The default speech repo is `Zoont/faster-whisper-large-v3-turbo-int8-ct2`,
revision `846f74797293e5f605d509a6ab5395a1f2f2874e`. That exact revision was
read from the local Hub `refs/main` used by the previous design, not resolved
over the network for this change. Tests inject loaders and placeholder assets;
they do not validate real model weights or claim their availability/license.
Runtime defaults resolve into this project's ignored cache. An explicit local
reuse path is an optional override, not a dependency on the reference checkout.

External dependencies are installed, not redistributed in this repository:
SunFounder_PiCar (the vendor's `picar` package and bundled PCA9685 module),
RPi.GPIO, OS SMBus/I2C support, NumPy and OpenCV for the Pi; faster-whisper,
CTranslate2/PyAV and Hugging Face Hub for optional server STT. Installation
requirements are in `requirements-robot.txt` and server-only
`requirements-speech.txt`. Do not infer a
license grant for the sibling project, vendor source or model assets from this
functional port. Before physical deployment or redistribution, review and
retain the selected upstream LICENSE/NOTICE files and model-card terms,
including transitive native libraries. Vendor installation and physical
compatibility have not been tested.

### Verified Vendor Pin

The follow-up source-only network check resolved both upstream `HEAD` and
`refs/heads/master` with `git ls-remote` to
`9890ab7d11dbce10c4bb6f0bffe61b7b08783c5e`. Reading
[setup.py at that exact commit](https://github.com/sunfounder/SunFounder_PiCar/blob/9890ab7d11dbce10c4bb6f0bffe61b7b08783c5e/setup.py)
confirmed the declared distribution is `SunFounder_PiCar`, version `1.0.1`,
import package `picar`, and metadata license label `GNU`. The
[upstream LICENSE](https://github.com/sunfounder/SunFounder_PiCar/blob/9890ab7d11dbce10c4bb6f0bffe61b7b08783c5e/LICENSE)
contains the GNU GPL version 2 text. This identifies the checked files; it
does not assert a license for the sibling checkout or model weights.

The pinned vendor wheel, servo and config-reader sources were also read,
not imported. `Front_Wheels` keeps its effective offset on `wheel.offset`
and clamps electrical angles via `turning_max`; constructors otherwise
read a local `config` file. The local adapter now clears the actual servo
offset, uses the full electrical range before its own measured travel
limits, and passes `/dev/null` as the vendor config source. Injected tests
exercise these API details without running upstream code. No package was
installed and no device/model was accessed during this network check.

Software acceptance uses injected ports, recorded audio and fake model
loaders only. No hardware, microphone, speaker, downloads or physical
calibration/actuation were used or validated.

The simulator is newly written against installed PyBullet APIs, with native
primitive geometry and no vendored meshes, URDFs or external runtime assets.
PyBullet/Bullet is an external installed dependency, not copied project code;
its upstream license is [zlib](https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt).
The upstream [wheel/steering example](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/racecar.py)
was consulted to check motor-control APIs; its source was not copied.
