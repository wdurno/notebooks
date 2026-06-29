# Robot Demo Legacy Import

Source copied from:

`../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/`

Imported for:

1. Phase 1 data generation mechanics.
2. Speech-to-text and text-to-speech support.
3. Qwen/QLoRA model loading references.
4. Replay, snapshot, and observation persistence references.
5. SSR/EWC/Lanczos phase 3 source material.
6. Existing unit and integration tests.

The old single-VLM `t` interpolation architecture is not the target architecture for this repo.
Refactor useful mechanics into the current VLM/LSTM KL-projection design.

