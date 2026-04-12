# Online Core

This package implements the online-learning variant of the SSR design.

The goal is to keep the convenience of Fisher-based SSR regularization while
switching from cumulative replay-style memorization to true online updates:

- optimize immediately after a new observation is processed
- reuse the same forward/backward pass for both fitting and Fisher updates
- avoid replaying large VLM activations just to refresh sufficient statistics
- keep the Fisher estimate relevant with exponential forgetting rather than a
  long-running arithmetic average

## Motivation

The original `SSRAgent` design is strong for cumulative memorization, but it is
not the right abstraction for a genuinely online robot loop.

In this repository, the online core exists for settings where:

- observations arrive sequentially
- memory is tight, especially for VLM activations
- we want one gradient computation per new observation
- the model should adapt while the target distribution is drifting

The online design uses an EMA-updated Fisher estimate, motivated in
[`AGENTS.md`](./AGENTS.md), so recent information is weighted more heavily than
old information.

## Main Ideas

- `OnlineSSRAgent` inherits from `SSRAgent`, but owns its own online
  `memorize()`, `ssr()`, `optimal_pi()`, and `fit()` behavior.
- Its `fit()` method is intentionally narrow: concrete agents build the current
  scalar task loss themselves, then hand that loss to the framework so it can
  perform the standardized post-loss sequence of SSR mixing, backward,
  gradient caching, optional clipping, optimizer step, and EMA memorization.
- The Fisher estimate is stored in the same low-rank-plus-diagonal style as the
  base SSR machinery, but interpreted in EMA scale rather than cumulative
  sample-sum scale.
- Lanczos is still used as a compression engine, but the online path builds an
  EMA Fisher operator and a matching diagonal update before recompressing.
- The online agent tracks diagnostics such as:
  - total observed steps
  - most recent `pi`
  - squared EMA weight mass
  - effective sample size

## When To Use It

Use the online core when you already have:

- a model class that can produce a loss for one freshly collected transition
- a need to keep SSR regularization active during continual adaptation
- a preference for online updates over replay-heavy reprocessing

It is especially appropriate for robotics and personalized-control workflows
where the environment is changing and the agent should adapt on-device with
minimal wasted computation.

## Package Contents

- `online_ssr_agent.py`: abstract online SSR base class
- `AGENTS.md`: mathematical and implementation specification for the online core

## Relationship To The Model Package

The online core does not replace `src/model/`.
Instead, it provides the online sufficient-statistics layer that a concrete
model can inherit when it wants:

- SSR-style regularization
- online EMA Fisher updates
- efficient one-pass fit/memorize behavior

For the current demo, this is designed to pair naturally with the PiCar control
stack while preserving existing target-network update cadence and snapshot
loading behavior.
