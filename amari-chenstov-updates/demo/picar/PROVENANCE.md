# Source Inspection And Licensing

Read-only reference: the sibling `kl-projection` checkout, tracked by the
enclosing repository at commit `6d6c88af8f405b44957254c4fa6c1ff7d49ed854`.
The inspected modules matched that revision; paths below are relative to the
reference checkout's root. This is provenance, not a runtime dependency or
an installation path. The checkout had no discovered LICENSE/COPYING
file or license declaration in README/pyproject metadata at inspection.
Consequently no implementation code or hardware adapter was copied from it;
no redistribution license is asserted for its contents.

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

The simulator is newly written against installed PyBullet APIs, with native
primitive geometry and no vendored meshes, URDFs or external runtime assets.
PyBullet/Bullet is an external installed dependency, not copied project code;
its upstream license is [zlib](https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt).
The upstream [wheel/steering example](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/racecar.py)
was consulted to check motor-control APIs; its source was not copied.
