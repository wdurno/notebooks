# Plan 19 PiCar Experiment

See [the plan](plan19.md) for theory and the scientific contract, and
[the handoff](PLAN19_HANDOFF.md) for current implementation and acceptance
status. Phase 0 is software acceptance, not evidence of robot competence.

## Environment

Run commands from the repository root. A fresh checkout does not need another
project's code or model directory. Python 3.12 is the tested interpreter.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r numerical_experiment/picar/requirements-server.txt
```

Use `requirements.txt` instead for CPU smoke/simulation without Qwen, or
`requirements-report.txt` for artifact-only reporting. Choose a Torch/CUDA
wheel appropriate for the machine and check PyTorch CUDA availability before
GPU work. The tested host uses Torch 2.10.0+cu128; pinning Python packages does
not install an NVIDIA driver. A working existing venv can be used instead.

PyBullet 3.2.7 may need a source build on Python 3.12. Its upstream build
chooses compiler parallelism from CPU/RAM counts. The acceptance host limits
that build to two compiler workers; this does not modify physics algorithms.
Allow enough RAM/time for installation and do not run a large build alongside
a memory-constrained experiment without checking resources.

## Model Assets

Explicit Qwen loading downloads the pinned Hugging Face revision into
`cache/numerical_experiment/picar/models/` by default and retains it for reuse.
The exact revision and file checksums are recorded in run metadata. Imports,
CPU smoke tests, and notebook/reporting commands never download models.

An existing checkpoint directory may be supplied with `--model-path` to avoid
another download. This is an optional optimization, not a dependency on that
directory for other users. `--offline` forbids downloads and must fail if the
requested files are absent. Do not edit source to insert a personal model path.

## Boundaries

- All robot-side service code is in `demo/picar`; the Pi does not need Qwen or Torch.
- Shared interfaces and simulator code are in `src/robotics`.
- Shared numerical functions are in `src/continual/picar_math.py`.
- The runtime does not import the earlier `kl-projection` checkout.
- Generated artifacts and model weights belong in ignored `cache/` or an explicitly provided archive root.
- The physical robot remains disarmed by default. Software acceptance uses a fake driver.

## Acceptance Commands

Run the complete bounded CPU suite and paired smoke pipeline:

```bash
.venv/bin/python -m numerical_experiment.picar.acceptance \
  --output cache/numerical_experiment/picar/phase0/acceptance-v1
```

Add `--resume` after an interruption. Completed units are verified and skipped;
changing scientific source or configuration requires a new output directory.
The fake-service test binds an ephemeral loopback port, never a physical robot.

For actual quantized Qwen, use the separate bounded CUDA check. This command
downloads and retains the pinned weights when no existing path is supplied:

```bash
.venv/bin/python -m numerical_experiment.picar.qwen_acceptance \
  --output cache/numerical_experiment/picar/phase0/qwen-check-v1
```

Optionally append `--model-path <checkpoint-directory>`; use `--offline` when
network access is forbidden. GPU acceptance checks a small input, not the full
production replay/context envelope. The adapted suffix uses FP32 computation
over frozen 4-bit weights; the frozen lower prefix remains reduced-precision.

For a manually bounded smoke/restart check and artifact-only reporting:

```bash
.venv/bin/python -m numerical_experiment.picar.run smoke \
  --output cache/numerical_experiment/picar/phase0/smoke-v1 --max-events 3
.venv/bin/python -m numerical_experiment.picar.run smoke \
  --output cache/numerical_experiment/picar/phase0/smoke-v1 --resume
.venv/bin/python -m numerical_experiment.picar.run status \
  --output cache/numerical_experiment/picar/phase0
.venv/bin/python -m numerical_experiment.picar.run report \
  --output cache/numerical_experiment/picar/phase0
```

`run stop --output <run-directory>` requests a checkpoint-boundary halt. The
next explicit `run smoke --resume` clears that request. An external archive
is optional via `--archive-root <mounted-directory>`; loss of that mount must
stop writing, not silently redirect data onto the primary disk.

See the handoff for measured results and remaining calibration work. Do not
substitute the tiny model or fake teacher into a production condition when a
dependency is missing. Phase 0 does not authorize Phases 1--5 or robot motion.
