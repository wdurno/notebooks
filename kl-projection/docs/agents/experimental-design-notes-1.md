# Experimental Design Notes 1

## Abstract

These notes preserve design alignment for the KL-projection robotics experiment.
The project is a proof of concept for a cheap hierarchy: a small VLM fires sparsely, an LSTM acts every step, and later fine tuning attempts to make the system viable on limited hardware.
The experiment values practical feel, low cost, and product-relevant constraints over exhaustive academic benchmarking.

## Table of Contents

1. [Core Hypothesis](#core-hypothesis)
2. [Robot Task](#robot-task)
3. [Model Hierarchy](#model-hierarchy)
4. [Action Distributions and KL](#action-distributions-and-kl)
5. [Metrics](#metrics)
6. [Phase 1 Data Collection](#phase-1-data-collection)
7. [Phase 2 KL Projection](#phase-2-kl-projection)
8. [Phase 3 Finalization](#phase-3-finalization)
9. [Known Constraints](#known-constraints)
10. [Deferred Decisions](#deferred-decisions)
11. [Implementation Implications](#implementation-implications)

## Core Hypothesis

An existing high-quality transformer can cost-effectively seed a robotics model hierarchy.
The expensive model should supply cognition and intent.
The cheaper model should supply fast, repeated action.
The user is intentionally starting from open source weights and a single consumer GPU, because democratization and cost reduction are central goals.

## Robot Task

The robot is a PiCar-V.
Its Raspberry Pi runs only a Flask server.
AI computation runs on a PC with a consumer-grade GPU.

The robot converses with the experimenter and chases a red ball.
The robot is slow and physically harmless.
The experimenter can pick it up when it gets close to collision.
Do not overbuild safety mechanics unless later requested or required by the old codebase.

## Model Hierarchy

The VLM is `Qwen2.5-VL-3B`.
The LSTM acts every game step.
The VLM acts every `K` game steps.
`K` is a hyperparameter, but the user does not expect to tune it heavily for the proof of concept.

The VLM reads the prior `K` observations and emits a head vector once per `K`-step series.
That head vector is concatenated to the visual encodings given to the LSTM.
The head-vector portion of the LSTM input is constant for the next `K` steps.

There are no explicit subgoals in the head vector.
The user does not want to micromanage available strategies at this stage.
The head vector should be learned through phase 2 KL projection and phase 3 combined training.

The VLM owns language generation.
The LSTM owns action generation.
The LSTM does not communicate back to the VLM with custom tokens, because that would raise the compute budget.

## Action Distributions and KL

Both the VLM and LSTM will emit probability distributions over actions.
The VLM-only system chooses from a small list of agentic actions.
Those agentic actions are deterministically converted into action probability vectors.
In phase 1, many VLM action probabilities may be almost sure probabilities, effectively 1 or 0.

Do not decide yet whether to smooth, temper, or otherwise soften these teacher distributions.
The user wants to revisit that after showing the old codebase.

The old codebase already constrains the action set.
It is not arbitrary Python generation, despite README shorthand about agentic Python commands.

## Metrics

Two recurring metrics matter across phases:

1. Action latency.
2. Coherency.

Action latency should be recorded as raw wall clock statistics.
Do not prematurely collapse latency into only mean, p50, p90, or another aggregate.
Aggregation will be decided later during results write-up.

Coherency means the robot's ability to do as it says.
Prior experiments used the original VLM under QLoRA parameters to evaluate whether actions align with verbal commands.
The same style of VLM-based scoring can be used here.
The user already observed this signal with the old codebase.
`Qwen2.5-VL-3B` is noisy and somewhat weak, but it is the best currently available model under the hardware constraints.
If the experiment is marginally successful, the user may procure hardware and move to 8B or larger models.

## Phase 1 Data Collection

Phase 1 gives the VLM complete control without involving the LSTM.
The purpose is data collection for KL projection.
The user already has an offline dataset from an old codebase, so phase 1 is effectively close to done.

Expected logs include images, text commands from the experimenter, text responses from the robot, and action probability vectors.
Most game state transitions may contain only images and action probability vectors.

Primary metrics are dataset size, action latency, and coherency.

## Phase 2 KL Projection

Phase 2 fits the LSTM from phase 1 data using KL projection.
At minimum, phase 2 needs a loss function.
Additional offline metrics can be added to describe learning success.

Offline replay over the existing dataset is worthwhile before robot runs.
This is not full simulation.
It is a cheap check that the LSTM learns meaningful action distributions from recorded observations and teacher vectors.

After KL projection, the full hierarchy runs on the robot.
The LSTM controls movement.
The VLM controls language and high-level strategy through its additional head.

The desired proof-of-concept result is visible speed-up plus adequate coherency.
Perfect coordination is not required.
The goal is to get something good enough to justify phase 3.

## Phase 3 Finalization

Phase 3 is intentionally ambitious and exploratory.
It exists because phases 1 and 2 alone are not the final product target.
The goal is cheap, fast, fine-tuned, small-data robotics.
Phase 3 is the attempt to break into a viable product specification.

Fine tuning options should be configurable independently where possible:

1. QLoRA rank of 0 disables VLM QLoRA parameters.
2. EWC regularizer weight `lambda = 0` disables EWC.
3. Replay buffer maximum length controls experience replay capacity.
4. The RL loss is the Actor-Critic baseline and cannot be disabled.

EWC is expected to be tricky on limited hardware, especially online consolidation.
Still, it remains part of the target product specification because it supports democratized, lower-cost continual learning.

DAgger-style corrective data collection is a good phase 3 candidate.
Let the LSTM act, periodically get corrective labels from the VLM or experimenter, and add those states to training.

## Known Constraints

The VLM visual stack is considered fast enough for this experiment.
Do not assume vision encoding latency is the dominant blocker unless measurements show it.

The coherency judge is noisy because `Qwen2.5-VL-3B` is limited.
Do not treat this as a reason to abandon the metric.
It is the currently available signal.

The user does not want expensive side metrics that add substantial experimenter workload.
Favor metrics already produced by the system: raw latency, losses, action distributions, text logs, VLM coherency scores, and recorded observations.

This is a proof of concept.
The experiment is judged by visible speed-up and sufficient coherency, not exhaustive optimization of `K` or a full benchmark suite.

## Deferred Decisions

Wait for the old codebase before deciding:

1. Whether teacher action distributions need smoothing.
2. Exact Actor-Critic action space details.
3. Exact offline phase 2 metrics beyond the KL loss.
4. Exact coherency scoring prompt and output schema.
5. Exact software architecture.

## Implementation Implications

Keep phase 3 knobs easy to disable or set to no-op values.
Record raw latency events with enough detail to aggregate later.
Preserve images, action probability vectors, experimenter text, and robot text in the dataset format.
Keep the action API small and structured.
Treat the VLM head vector as a learned conditioning vector, not a hand-authored subgoal planner.
Design data logging so offline replay and robot runs share as much machinery as practical.
