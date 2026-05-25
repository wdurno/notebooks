# Implementation strategy details 

`README.md` is a human-readable overview of this work. 
This document covers implementation instructions in-detail.

You, the agent assistant, should add to this file to maintain implementation notes. 

This experiment will be written in Pytorch. 

Maintain a `.gitignore` file in the root to keep the repo clean.

Models and training data will be stored in `data/`. 
Use `.gitignore` to avoid accidentally uploading files under this directory to GitHub. 
They are too large for that. 

Maintain implementation code in an `src/` directory. 

Maintain experimental `.ipynb` notebooks in an `experiments/` directory.

## Documentation split

`README.md` should stay short, human-readable, and written as the project summary.
Use this file for implementation decisions, experiment contracts, dataset details, metric definitions, and other notes needed by the coding assistant.

This is a non-standard use of `AGENTS.md`, because many projects use it as a user-authored harness for agents rather than as a project design log.
For this repo, the convention is explicit:
1. User intent and durable implementation instructions belong here.
2. Human-facing motivation and results belong in `README.md`.
3. Highly structured experiment configuration should eventually move into config files under `configs/` if it becomes too detailed for prose.

## Data generation

Generate custom JSONL datasets under `data/generated/`.
Do not commit generated data.

Each example should include enough metadata to support exact evaluation and experiment slicing:

```json
{
  "id": "task_a_train_000001",
  "task_family": "factory_qa",
  "task_id": "factory_a",
  "split": "train",
  "prompt": "...",
  "target": "...",
  "answer_key": "...",
  "metadata": {
    "entity_id": "ZX-17",
    "rule_id": "coolant_lookup",
    "requires_reasoning": false
  }
}
```

Use these split names:
1. `ewc_init`: observations used to estimate the EWC precision structure before learning a later task.
2. `train`: observations used to fine tune QLoRA parameters on the current task.
3. `eval_seen`: held-out prompts over entities and rules observed in training.
4. `eval_heldout`: held-out prompts over entities or compositions not observed in training.
5. `eval_original`: prompts intended to measure retention of original base-model behavior not introduced by local fine tuning.

### Dataset families

Start with deterministic, synthetic datasets where exact scoring is possible.

1. `factory_qa`: generate a fake factory manual with machines, components, error codes, materials, thresholds, and safety rules.
   Include direct lookup questions, two-hop questions, and safety decision questions.
   Example target behavior: answer the required field exactly, or output a constrained decision such as `ALLOW_RESTART` or `DENY_RESTART`.

2. `rule_transform`: generate symbolic input-output transformations.
   Include routing-code transforms, label normalization, and structured string conversions.
   Make some task pairs partially conflicting so forgetting is measurable.
   Example: Task A maps `AB-17` to `ROUTE_BLUE_17`; Task B maps `AB-17` to `ROUTE_RED_17`.

3. `preference_following`: generate user preference instructions with measurable constraints.
   Include dietary exclusions, units, length preference, schedule windows, and output format rules.
   Prefer constrained outputs at first so `target_format_validity` and exact checks are meaningful.

### Base-model retention data

Do not treat the EWC initialization dataset as the only source of "old knowledge."
We also need `eval_original` to estimate retention of information or behaviors that were already present in `Phi-4-mini`.

Recommended approach:
1. Generate candidate general prompts with code or another model.
2. Query the unfine-tuned `Phi-4-mini` for reference answers.
3. Filter to examples where the base model answers consistently under low-temperature decoding.
4. Store prompts, reference answers, and loose answer validators.
5. Never train on `eval_original`.

This lets forgetting metrics distinguish loss of base-model behavior from forgetting only the immediately previous synthetic task.

## EWC setup

The experiment generalizes diagonal EWC by modeling a positive semidefinite precision matrix as:

```text
P = L L^T + Lambda
```

where:
1. `L` has shape `[num_trainable_parameters, ewc_rank]`.
2. `Lambda` is a non-negative diagonal vector.
3. `EWC rank` is the number of columns in `L`.
4. `EWC n0` is the number of observations used to estimate `L` and `Lambda`.

Estimate EWC over trainable QLoRA parameters first.
Keep frozen base-model parameters out of the EWC penalty unless an experiment explicitly changes that.

For a parameter vector `theta`, saved reference vector `theta_star`, and `delta = theta - theta_star`, the rank-generalized EWC penalty is:

```text
ewc_penalty = lambda_ewc * (||L^T delta||_2^2 + sum_i Lambda_i * delta_i^2)
```

The corresponding drift diagnostic is:

```text
ewc_weighted_drift = ||L^T delta||_2^2 + sum_i Lambda_i * delta_i^2
```

Use this instead of the diagonal-only `sum_i F_i * delta_i^2` whenever `ewc_rank > 0`.

## Metrics

Use exact, low-variance metrics first.

1. `Target EM`: exact match on the current task's evaluation set.
   Normalize whitespace and casing only when the dataset generator declares that safe.

2. `target_token_f1`: token-level F1 between generated answer and target answer.
   This is useful when the answer can contain multiple fields and exact string equality is too brittle.
   It should be secondary to exact match for constrained synthetic tasks.

3. `target_format_validity`: fraction of generated outputs satisfying the required schema or output grammar.
   Examples: valid JSON, one of an allowed label set, contains required keys, or matches a generated regex.

4. `retention_score`: score on a previous task or on `eval_original` after new-task fine tuning.

5. `forgetting_delta`: `score_before_new_task - score_after_new_task`.

6. `retention_ratio`: `score_after_new_task / max(score_before_new_task, epsilon)`.

7. `average_retention`: average retention ratio across all prior tasks and `eval_original`.

8. `worst_task_retention`: minimum retention ratio across all prior tasks and `eval_original`.

9. `fisher_similarity`: similarity between a small-`EWC n0` precision estimate and a larger reference estimate.
   For diagonal-only EWC, use cosine similarity between diagonal vectors.
   For rank-generalized EWC, compare matrix-vector products on a fixed set of probe vectors, or compare the top subspaces of `L`.

10. Hardware metrics: `peak_vram_mb`, `train_wall_seconds`, `fisher_wall_seconds`, `tokens_per_second`, `adapter_disk_mb`, and `replay_buffer_size_mb`.

### Case 1 conflict-aware retention metrics

For `rule_transform` Case 1, Task A and Task B intentionally conflict only on the routing rules for prefixes `AB`, `EF`, and `JK`.
Task A maps those prefixes to `BLUE`; Task B maps them to `RED`.
Prefixes `CD`, `GH`, and `LM` remain `GREEN` in both tasks.

Track Task A retention both in aggregate and by conflict group:
1. `task_a_conflicting_before_em`.
2. `task_a_conflicting_after_em`.
3. `task_a_nonconflicting_before_em`.
4. `task_a_nonconflicting_after_em`.
5. `task_a_conflicting_n`.
6. `task_a_nonconflicting_n`.

This prevents the aggregate `task_a_after_em` from hiding whether the model retains shared rules while forgetting only rules overwritten by Task B.

## Experimental cases

### Case 1: EWC n0 sweep

Goal: measure how many observations are needed to initialize a useful EWC regularizer.

Fixed:
1. Base model: `Phi-4-mini`.
2. Hardware: `NVIDIA GeForce RTX 4070`.
3. Task sequence: Task A then Task B.
4. Experience replay buffer: `0`.

Sweep:
1. `EWC n0`: `0, 4, 8, 16, 32, 64, 128, 256, 512`.
2. `EWC rank`: `0, 1, 2, 4, 8`.
3. `EWC lambda`: `0.1, 1.0, 10.0, 100.0`.

Measure:
1. Target EM on Task B.
2. Retention on Task A.
3. Retention on `eval_original`.
4. Fisher similarity to a larger-`n0` reference estimate.
5. Hardware metrics.

### Case 2: EWC versus experience replay

Goal: find trade-offs between EWC initialization quality, EWC rank, and replay buffer size.

Sweep:
1. `EWC n0`: `0, 16, 64, 256`.
2. `EWC rank`: `0, 2, 4, 8`.
3. `ER buffer`: `0, 8, 16, 32, 64, 128, 256`.

Measure target learning, prior-task retention, base-model retention, and hardware cost.

### Case 3: QLoRA rank versus EWC rank

Goal: test whether larger adapters increase plasticity without unacceptable forgetting, and whether higher EWC rank controls that drift.

Sweep:
1. `QLoRA rank`: `4, 8, 16, 32`.
2. `EWC rank`: `0, 1, 2, 4, 8`.
3. `EWC n0`: `16, 64, 256`.

Measure target learning, prior-task retention, base-model retention, parameter drift norm, and `ewc_weighted_drift`.

### Case 4: Fine tuning dataset size sweep

Goal: separate poor EWC initialization from insufficient new-task data.

Sweep:
1. `Train n`: `8, 16, 32, 64, 128, 256, 512`.
2. `EWC n0`: `16, 64, 256`.
3. `EWC rank`: `0, 2, 4, 8`.

Measure target learning, retention, and held-out generalization.

### Case 5: Multi-step continual learning

Goal: discover which contexts allow new learning and retention to coexist under small data and small compute.

Sequence:
1. `factory_qa`.
2. `rule_transform`.
3. `factory_safety_decision`.
4. `preference_following`.

Evaluate after every step against the current task, every previous task, and `eval_original`.
Use `average_retention` and `worst_task_retention` to reveal whether performance is broadly retained or whether one context is sacrificed.

## Build Phase 1

Goal: implement the first end-to-end experimental path for Case 1 while keeping each piece testable on small synthetic data before downloading or training the full model.

### Phase 1.1: repository scaffold

Create the implementation structure:
1. `src/` for Python modules.
2. `configs/` for experiment and model configuration.
3. `experiments/` for notebooks.
4. `outputs/` for run logs, metrics, and generated reports.
5. `data/` for downloaded models and generated datasets.

Use the Python virtual environment at `~/.venv`.
Maintain dependencies in `requirements.txt`.

### Phase 1.2: deterministic synthetic data

Implement dataset generation first.

Required dataset families for Phase 1:
1. `factory_qa`.
2. `rule_transform`.

Defer `preference_following` until the first model-training path is working.

Generate JSONL examples with split labels:
1. `ewc_init`.
2. `train`.
3. `eval_seen`.
4. `eval_heldout`.

Make outputs constrained enough for exact-match scoring.
Include mild conflicts between Task A and Task B so retention pressure is measurable.

### Phase 1.3: metrics

Implement metrics independent of model training:
1. `target_exact_match`.
2. `target_token_f1`.
3. `target_format_validity`.
4. `retention_score`.
5. `forgetting_delta`.
6. `retention_ratio`.
7. `average_retention`.
8. `worst_task_retention`.

Use deterministic unit-scale checks against generated examples before running GPU training.

### Phase 1.4: model loading and QLoRA

Load `microsoft/Phi-4-mini-instruct` through Hugging Face.
Store downloaded model artifacts under `data/`.

Training defaults:
1. 4-bit quantization with `bitsandbytes`.
2. Small per-device batch size.
3. Gradient accumulation.
4. Gradient checkpointing.
5. FlashAttention when available.
6. Short synthetic sequence lengths for the first runs.

Only train LoRA adapter parameters.

### Phase 1.5: EWC implementation

Implement diagonal EWC and rank-generalized EWC over trainable QLoRA parameters.

Use `src/lanczos.py` as the local Lanczos implementation.
Do not import or modify the original source repo.

Lanczos implementation decisions:
1. Remove damping/`eps` features entirely.
2. Keep computations on the requested torch device.
3. Avoid unnecessary CPU tensor construction such as `torch.tensor(list_of_tensors)`.
4. Average Fisher-vector products by sample count so `EWC n0` does not silently rescale the regularizer.
5. Stop early when the Krylov residual norm is at or below `min_off_diag`, returning the lower-rank approximation produced so far.

For zero or tiny off-diagonal values, the intended guardrail is early termination.
This indicates the current Krylov space has exhausted meaningful residual direction under the estimated Fisher operator.
Do not divide by the tiny value.

### Phase 1.6: original-model retention machinery

Implement after the basic synthetic Case 1 path works, but keep it inside Phase 1.

Steps:
1. Generate a candidate bank of general prompts.
2. Query the unfine-tuned `microsoft/Phi-4-mini-instruct` with deterministic or low-temperature decoding.
3. Repeat generation for each prompt and keep only stable responses.
4. Store accepted prompts as `eval_original` JSONL examples.
5. Attach validators: exact match for short factual/stable answers, token F1 for mildly variable answers, and regex/schema checks for constrained outputs.
6. Never include `eval_original` in `ewc_init`, `train`, or experience replay unless a later experiment explicitly tests that condition.

This measures whether local fine tuning preserves behavior that was already present in the base model rather than only preserving information introduced by synthetic Task A.

### Phase 1.7: Case 1 runner

Implement a CLI runner for:
1. Train Task A.
2. Estimate EWC from Task A `ewc_init` using the selected `EWC n0` and `EWC rank`.
3. Train Task B with the EWC penalty.
4. Evaluate Task B learning.
5. Evaluate Task A retention.
6. Evaluate `eval_original` retention when available.
7. Write one JSONL metrics record per run under `outputs/`.

Initial sweep:
1. `EWC n0`: `0, 4, 8, 16, 32, 64`.
2. `EWC rank`: `0, 1, 2, 4`.
3. `EWC lambda`: `0.1, 1.0, 10.0`.

Expand to the full sweep after verifying runtime and VRAM on the RTX 4070.

### Phase 1.8: open concern

The Lanczos approach and the proposed gradient-matrix randomized SVD approach are materially similar when both use the same Fisher-vector product:

```text
F v = G^T G v / n
```

The practical distinction is implementation shape.
Lanczos works naturally with a matrix-free operator and avoids storing `G`.
Randomized SVD can be simpler to reason about when `EWC n0` and trainable parameter count are small enough to materialize per-example gradients.
For this repo, prefer Lanczos first because it better matches the target memory constraints and the existing implementation.

### Phase 1 implementation status

Implemented Phase 1 files:
1. `src/data/generate.py`: deterministic `factory_qa` and `rule_transform` JSONL generation.
2. `src/data/io.py`: JSONL loading, writing, and split filtering.
3. `src/metrics.py`: exact match, token F1, format validity, and retention helpers.
4. `src/modeling.py`: tokenizer, base-model loading, and QLoRA model loading.
5. `src/training.py`: supervised fine tuning loop, generation, and evaluation.
6. `src/ewc.py`: trainable-parameter flattening, Fisher estimation, diagonal EWC, rank-generalized EWC, and EWC penalty.
7. `src/original_retention.py`: base-model stable-response retention set generation.
8. `src/run_case1.py`: Case 1 CLI runner and metrics logging.
9. `configs/case1.yaml`: initial Case 1 configuration.
10. `experiments/build_phase_1_case1.ipynb`: notebook entry point for dry checks and optional GPU runs.

Verified without downloading the model:
1. Python modules compile.
2. Notebook JSON parses.
3. Synthetic data generation writes JSONL.
4. Metric helpers run on deterministic examples.
5. Lanczos returns finite low-rank and residual diagonal outputs on a tiny gradient smoke test.

Still requires GPU/model execution:
1. Install dependencies in `~/.venv`.
2. Download `microsoft/Phi-4-mini-instruct` into `data/models`.
3. Build `eval_original` from stable base-model generations.
4. Run a one-combination Case 1 smoke job.
5. Expand to the configured sweep after VRAM and runtime are measured.

Recent implementation note:
1. Future prediction artifacts include example metadata.
2. Future Case 1 metrics include conflict-aware Task A retention fields.
3. `experiments/build_phase_1_case1.ipynb` also derives conflict-aware retention from existing prediction artifacts by joining predictions back to generated examples, so previously generated runs can still be interpreted.
