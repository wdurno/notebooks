# Phase A Import Provenance

## Summary

Build Phase A copied relevant assets from prior repositories into this repo.
Old repositories should not be modified.
Future changes should happen in this repo.

## Copied Assets

1. Robot-demo source.
   - Source: `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/src/`
   - Destination: `/src/picar_kl/legacy/robot_demo/src/`
   - Status: copied as staged legacy source, excluding bytecode caches.

2. Robot-demo tests.
   - Source: `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/tests/`
   - Destination: `/src/picar_kl/legacy/robot_demo/tests/`
   - Status: copied as staged legacy tests, excluding bytecode caches.

3. Robot-demo integration tests.
   - Source: `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/tests/integration/`
   - Destination: `/tests/integration/robot_demo/`
   - Status: copied and import paths adjusted to use staged legacy source.

4. Legacy robot-demo data.
   - Source: `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/data/`
   - Destination: `/artifacts/data/legacy_robot_demo/`
   - Status: copied locally, gitignored as generated data.

5. Phase 1 data subset.
   - Source: `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/data/phase1/`
   - Destination: `/artifacts/data/phase1/`
   - Status: copied locally, gitignored as generated data. This is a convenience subset; the full legacy data tree is preserved under `/artifacts/data/legacy_robot_demo/`.

6. Model manifests.
   - Source: `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/model/manifests/`
   - Destination: `/artifacts/manifests/tracked/models/`
   - Status: copied as tracked lightweight defaults.

7. PiCar API code.
   - Source: `../../picar-v-rl-env/src/car_env/`
   - Destination: `/src/picar_kl/robot/legacy_car_env/`
   - Status: copied as staged robot-side code, excluding bytecode caches.

8. PiCar API runner.
   - Source: `../../picar-v-rl-env/run_api.py`
   - Destination: `/src/picar_kl/robot/legacy_run_api.py`
   - Status: copied as staged runner reference.

9. Mathematical notebook.
   - Source: `../amari-chenstov-updates/mathematical_overview.ipynb`
   - Destination: `/notebooks/imported/amari_chenstov_mathematical_overview.ipynb`
   - Status: copied and contextualized by `/docs/humans/math-context.md`.

## Notes

The staged legacy source is not the final architecture.
Phase B should extract stable action, data, robot-client, and latency interfaces from these copies.
Imported data is preserved separately from new phase 1 data so loaders can support both old runs and fresh runs without hiding schema differences.
