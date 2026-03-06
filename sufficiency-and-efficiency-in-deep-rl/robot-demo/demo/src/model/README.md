# Model Package

This package implements requirement 2 from `demo/spec.md`: a PiCar control
model that combines a frozen Qwen 2.5-VL base model, LoRA adapters, and a
continuous actor-critic control stack inside the `SSRAgent` training framework.

## Design goals

- Keep the base VLM frozen and read-mostly.
- Use QLoRA-style 4-bit quantization (NF4) for the base VLM to reduce GPU memory.
- Train only LoRA parameters plus the actor/critic control heads.
- Keep agentic action selection available during the mixed-control phase.
- Learn robot control directly in the 4-key PiCar vector space used by
  `apply_vector`.
- Let the model own action interpolation by `t`, so the environment only sends
  observations and executes returned vectors.
- Keep unit tests independent from heavyweight model downloads by using a fake
  backbone.

## Package layout

### `config.py`

Top-level model configuration, including:

- VLM model location under `demo/model/`
- optimizer settings
- LoRA settings
- loss weights `alpha` and `beta`
- RL discount `gamma`

### `schemas.py`

Dataclasses that define the model/environment contract:

- `ModelObservation`: one environment observation
- `ModelActionOutput`: one model action result
- `Transition`: one replay-buffer record
- `TransitionBatch`: sampled training batch

### `action_space.py`

Defines the 8 agentic actions and the deterministic mapping from those actions
into PiCar vector dictionaries with keys:

- `pan`
- `tilt`
- `turn`
- `drive`

This module also handles:

- action-name/index conversion
- one-hot encoding
- one-hot to vector conversion
- vector-to-tensor conversion
- vector/tensor clamping to PiCar control bounds
- vector interpolation by `t`

### `replay_buffer.py`

Implements a model-local replay buffer for multimodal transitions. It is shaped
to work with `SSRAgent` without modifying the read-only code in `demo/src/core`.

### `backbones.py`

Backbone abstraction layer.

- `QwenLoRABackbone`: Hugging Face + PEFT + bitsandbytes implementation for
  a frozen 4-bit QLoRA base VLM with LoRA adapters
- `FakeBackbone`: deterministic lightweight backbone used by unit tests

The backbone is responsible for:

- turning observations into fused hidden states
- producing an agentic action choice
- optionally producing a VLM loss term
- optionally generating text for TTS
- switching to a text-only generation prompt once `t = 1`

### `picar_agent.py`

Defines `PiCarActionModel`, the main model class.

Responsibilities:

- inherit from `SSRAgent`
- run inference through `forward()`
- produce both the agentic and actor-driven actions
- mix those vectors by `t` to form the executed action
- score continuous state-action pairs with a single critic
- define the combined loss:
  - `0.5 * VLM loss`
  - `0.5 * RL loss`

## Model / environment contract

The environment should pass one `ModelObservation` per step:

- `image_rgb`: current RGB frame
- `messages`: Qwen-style history and user context
- `t`: interpolation coefficient in `[0, 1]`
- optional metadata like reward, done flag, and step index

The model returns one `ModelActionOutput`:

- `agentic_action_name`
- `agentic_action_one_hot`
- `agentic_action_vector`
- `actor_action_vector`
- `executed_action_vector`
- `critic_value`
- `generated_text`

The environment should:

1. send `executed_action_vector` to the PiCar API; and
2. if `generated_text` is non-empty, speak it with the speech subsystem.

## Training flow

`PiCarActionModel.loss()` uses `forward()` outputs and replay-buffer transitions
to compute:

- `L_vlm`: supervised VLM/tool-call loss
- `L_rl`: continuous actor-critic loss

with the default combined objective:

```text
L = 0.5 * L_vlm + 0.5 * L_rl
```

The RL path uses detached agentic action vectors, a Huber critic loss, and a
bootstrap target built from target actor/critic heads.

## Usage sketch

```python
from model import ModelConfig, ModelObservation, PiCarActionModel, TransitionReplayBuffer

replay_buffer = TransitionReplayBuffer(capacity=1024)
model = PiCarActionModel(replay_buffer=replay_buffer, config=ModelConfig())

observation = ModelObservation(
    image_rgb=frame_rgb,
    messages=messages,
    t=0.2,
)

action = model.forward(observation)
vector = action.executed_action_vector
```

## Dependencies

Unit tests use the fake backbone and only need `torch`.

Real Qwen-backed usage additionally expects:

- `transformers`
- `peft`
- `bitsandbytes`
- `Pillow`
- `huggingface_hub`

Real QLoRA-backed usage also requires CUDA-capable hardware.

The base model is expected under `demo/model/vlm/qwen2.5-vl-3b/base/`, with
auto-download support controlled by the VLM manifest.
