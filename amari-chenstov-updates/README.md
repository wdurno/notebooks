# Linearized Fisher Updates for Continual Learning

This project studies continual fine-tuning when new data, memory, and compute
are scarce. The motivating application is a pretrained model that must learn a
new task online while retaining useful information from its prior training.

Elastic Weight Consolidation (EWC) compresses old likelihood information into
an anchor and a Fisher information estimate. As the model moves, that Fisher
estimate also changes. The project therefore asks two related questions:

1. Can a linearized Fisher update (LFU), including the Amari-Chentsov and
   residual tensors, improve a recursively maintained Fisher summary?
2. Can the new-information weight $\pi_t$ be recommended online when repeated
   tuning trials are unavailable?

The mathematical construction, assumptions, and terminology are developed in
[mathematical_overview.ipynb](mathematical_overview.ipynb).

## Experimental setting

The controlled experiment gradually changes the proportion of digit 9 in an
MNIST stream. This moves the population solution along an implicit path

$$
p\longmapsto\theta^\star(p)\longmapsto
\mathcal I(\theta^\star(p)).
$$

Small neural networks, low per-step sample counts, bounded replay, and
rank-8-plus-diagonal Fisher summaries model the intended hardware- and
data-constrained regime. Computational work runs in immutable Python entry
points; notebooks only read completed artifacts and render analyses.

## Current evidence

- A direct exponentially moving averaged Fisher summary without LFUs is the
  practical baseline. The tested LFU corrections were noisy, frequently
  indefinite before projection, and did not justify their additional cost.
- Fixed $\pi=.05$ is the incumbent on the tested MNIST paths, not a general
  theorem favoring fixed composition. Hybrid EWC with a replay buffer of 32
  observations is the strongest tested constrained-memory strategy.
- Exponentially discounted risk (EDR) produces a structured online
  recommendation and can move away from its cold-start value. Its current
  closed-loop application underperforms the fixed MNIST comparators.
- The prequential statistic $C_t$ provides a useful single-trajectory alarm
  for stale residual-risk calibration. It does not certify predictive policy
  value.

Whether adaptive $\pi_t$ helps in applications where the locally appropriate
composition changes materially and repeated tuning trials are unavailable
remains open.

## Repository map

- `mnist-findings.ipynb`: concise experimental synthesis with artifact-backed
  figures and evidence strength.
- `mathematical_overview.ipynb`: concise theory with rigorous appendices.
- `mnist_experiment/`: controlled experiments, plans, conditions, and results
  notebooks.
- `src/`: shared estimators, matrix representations, controllers, and artifact
  handling.
- `test/`: deterministic unit and integration tests.
- `demo/`: applied demonstrations built only from findings that survive the
  controlled experiments.
- `cache/`: generated datasets and immutable run artifacts; excluded from git.

See [AGENTS.md](AGENTS.md) and
[mnist_experiment/AGENTS.md](mnist_experiment/AGENTS.md) for the scientific and
software contracts.

## Citations

[1] Y. Zheng, Y. Zhang, J. van de Weijer, G. M. van de Ven, S. Du, X. Zhang,
and Z. Tian, [*Revisiting Weight Regularization for Low-Rank Continual
Learning*](https://arxiv.org/abs/2602.17559), arXiv:2602.17559, 2026.

[2] S. Amari and H. Nagaoka, *Methods of Information Geometry*, American
Mathematical Society, 2000.
