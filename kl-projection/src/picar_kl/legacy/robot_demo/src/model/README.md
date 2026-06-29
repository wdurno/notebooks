# Model Package

This package implements requirement 2 from `demo/spec.md`: a PiCar control
model that combines a frozen Qwen 2.5-VL base model, LoRA adapters, and a
continuous actor-critic control stack inside the `SSRAgent` training framework.

## Design goals

- Keep the base VLM frozen and read-mostly.
- Use QLoRA-style 4-bit quantization (NF4) for the base VLM to reduce GPU memory.
- Train only LoRA parameters plus the actor/critic control heads.
- Reuse the same quantized base Qwen instance for both policy generation and
  reward prompting; reward scoring runs in frozen inference mode with adapters
  disabled.
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
- generation settings (default `temperature=0.3`)
- deterministic decoding control (`deterministic_coding`)
- loss-term weighting controls
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
- define the combined objective across:
  - supervised VLM loss
  - continuous control RL loss
  - off-policy token policy-gradient loss
  - value-baseline regression loss
- expose explicit mode switches:
  - `set_inference_mode()` for rollout sampling/generation
  - `set_optimization_mode()` for fit/memorize phases

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
- `logp_beta_sum`

The environment should:

1. send `executed_action_vector` to the PiCar API; and
2. if `generated_text` is non-empty, speak it with the speech subsystem.

## Training flow

`PiCarActionModel.loss()` uses `forward()` outputs and replay-buffer transitions
to compute:

- `L_vlm`: supervised VLM/tool-call loss
- `L_rl`: continuous actor-critic control loss
- `L_pg`: off-policy token policy-gradient loss
- `L_value`: value-baseline regression loss

with the default combined objective:

```text
L = L_vlm + L_rl + L_pg + L_value
```

The replay buffer stores one behavior-policy sequence log-probability per
transition (`logp_beta_sum`) for off-policy importance weighting.

## Loss details

For a minibatch of size $B$, let:

- $s_i$: current hidden state for sample $i$
- $s'_i$: next hidden state for sample $i$
- $r_i$: reward
- $d_i \in \{0,1\}$: done flag
- $t_i \in [0,1]$: interpolation coefficient
- $a^{\text{exec}}_i$: executed action from replay (used for critic TD fit)
- $a^{\text{agentic}}_i$: agentic action vector from the backbone
- $a^{\text{actor}}_i$: actor-head action vector
- $\bar a^{\text{actor}}_i$, $\bar Q$: target actor and target critic
- $\log P_{\beta,i}$: stored behavior-policy sequence log-probability (`logp_beta_sum`)
- $\log P_{\theta,i}$: current-policy sequence log-probability of the replayed text

The objective is:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{vlm}}
+
\mathcal{L}_{\text{actor}}
+
\mathcal{L}_{\text{critic}}
+
\mathcal{L}_{\text{pg}}
+
\mathcal{L}_{\text{value}}.
$$

The TD target for the critic is:

$$
y_i
=
r_i + \gamma (1-d_i)\,\bar Q\!\left(
s'_i,\;
\operatorname{mix}\!\left(
\operatorname{detach}(a^{\text{agentic}\prime}_i),
\bar a^{\text{actor}}_i,
t_i'
\right)
\right).
$$

The critic loss is Huber (smooth L1):

$$
\mathcal{L}_{\text{critic}}
=
\frac{1}{B}\sum_{i=1}^{B}
\operatorname{Huber}\!\left(
Q(s_i, a^{\text{exec}}_i)-y_i
\right).
$$

For actor optimization, the mixed action is:

$$
\tilde a_i
=
\operatorname{mix}\!\left(
\operatorname{detach}(a^{\text{agentic}}_i),
a^{\text{actor}}_i,
t_i
\right).
$$

The anchor term penalizes actor drift from the detached agentic prior:

$$
p_i = \left\|a^{\text{actor}}_i-\operatorname{detach}(a^{\text{agentic}}_i)\right\|_2^2,
\qquad
w_i = \lambda_{\text{anchor}}(1-t_i),
$$

where $\lambda_{\text{anchor}}=\texttt{actor\_anchor\_weight}$.

The actor loss is:

$$
\mathcal{L}_{\text{actor}}
=
-\frac{1}{B}\sum_{i=1}^{B} Q(s_i,\tilde a_i)
+
\frac{1}{B}\sum_{i=1}^{B} w_i\,p_i.
$$

For the value baseline:

$$
v_i = V(s_i),\qquad
v_i^{\text{target}} = r_i + \gamma(1-d_i)V(s'_i),
$$

$$
\mathcal{L}_{\text{value}}
=
\frac{1}{B}\sum_{i=1}^{B}\operatorname{Huber}\!\left(v_i-v_i^{\text{target}}\right),
\qquad
\hat A_i = \operatorname{detach}\!\left(v_i^{\text{target}}-v_i\right).
$$

The off-policy token-PG term uses sequence-level importance ratios:

$$
\rho_i=\exp(\log P_{\theta,i}-\log P_{\beta,i}),
$$

$$
\mathcal{L}_{\text{pg}}
=
-\frac{1}{B}\sum_{i=1}^{B}
\rho_i\hat A_i
\sum_{k}\log\pi_\theta(a_{i,k}\mid a_{i,<k}, s_i).
$$

Implementation note: critic parameters are temporarily frozen when computing the
actor value term, so this term updates actor/backbone paths but not critic
weights.

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
