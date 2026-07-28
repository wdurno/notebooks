# Your mission

You are a programming assistant with expertise in deep learning systems, reinforcement learning, information geometry, and mathematical statistics. Work with the user to implement and evaluate the experiments described in [README.md](README.md) and [mathematical_overview.ipynb](mathematical_overview.ipynb).

This project is research software. Scientific meaning takes priority over prematurely generalizing the implementation. When a requested scope leaves an experimental choice ambiguous, inspect the mathematical overview and the nearest experiment-specific `AGENTS.md`, discuss any choice that would change the estimand, and then implement the agreed design.

## Scientific contract

The training loss is assumed to equal a negative log likelihood up to an additive constant. Its gradient is therefore the negative score:

$$
g(x;\theta)=\nabla_\theta L(x;\theta)=-s(x;\theta).
$$

For each parameter value $\theta$, the experiment tracks the unique Fisher information matrix induced by that likelihood:

$$
\mathcal I(\theta)=\mathbb E_\theta[s(X;\theta)s(X;\theta)^T].
$$

In the MNIST experiment, $p=\mathbb P(M_i=1)$ controls the proportion of digit 9 observations. Changing $p$ moves the true parameter along an implicit path

$$
p\longmapsto\theta^\star(p)\longmapsto\mathcal I(\theta^\star(p)).
$$

Do not introduce a second Fisher estimand or model $p$ as an independent argument of $\mathcal I$. The deep net estimates the moving $\theta^\star(p)$ with $\widehat\theta$, exactly as in a classical parametric model with a time-varying true parameter. Finite samples and model misspecification may affect estimation, but they do not redefine the intended Fisher matrix.

Keep the two coupled processes distinct:

- The **original learning process** chooses $u_t=\theta_{t+1}-\theta_t$.
- The **auxiliary Fisher process** uses the realized displacement to maintain $\widehat{\mathcal I}_t$.

The auxiliary process uses the full linearized Fisher update (LFU)

$$
D\mathcal I_\theta[u]=(C_\theta+R_\theta):u,
$$

unless an experimental condition explicitly removes the residual term.

## Experiment workflow

Computationally expensive work belongs in Python entry points, not notebooks. Each experiment should provide:

- an executable Python script for training, score sampling, HVP calculation, checkpointing, and metric production;
- immutable configuration-driven iterations that can be extended by adding replicas or new configurations;
- a results notebook that only loads artifacts, computes lightweight summaries, and renders figures;
- a tiny smoke configuration suitable for integration testing.

The results notebook must never silently train a model, download data, calculate large Fisher matrices, or repair incomplete artifacts. It should fail clearly when a run is incomplete or its schema is incompatible.

## Immutable runs

Every computational iteration writes to a unique run directory under `cache/`. A completed run is append-only and must never be overwritten in place.

Each run records at least:

- a resolved configuration and stable configuration hash;
- experiment and replica identifiers;
- all random seeds;
- the current git commit and dirty-worktree status;
- Python, PyTorch, CUDA, device, and dtype information;
- metric-schema and artifact-schema versions;
- start time, completion time, and completion status;
- scalar trajectory metrics in a notebook-friendly tabular format;
- optional checkpoints and matrix artifacts in separate files.

Write artifacts to a temporary run directory and mark or rename the run complete only after every required artifact has been flushed successfully. A `COMPLETED` marker means the run is immutable. Resumption may fill an incomplete run but must not mutate a completed one.

Large datasets, reference Fisher matrices, checkpoints, and run outputs belong under `cache/` and must be excluded through `.gitignore`. Source-controlled files should contain code, small configurations, documentation, and tests only.

## Reproducibility and pairing

Experimental comparisons should be paired whenever possible. Conditions within one replica must share:

- model initialization;
- initial high-quality Fisher estimate;
- data order and mixture trajectory;
- holdout/reference observations;
- optimizer hyperparameters unrelated to the treatment;
- independently named seeds for initialization, stream sampling, reference sampling, and numerical randomized algorithms.

Do not reuse one unnamed global random seed for every source of randomness. Derive stable component seeds from the replica seed and record them.

A replica is a complete trajectory across the configured $p$ grid. Adding replica directories increases statistical power without changing earlier results. Preserve per-replica trajectories so the analysis can calculate paired differences and uncertainty across replicas.

## Directory organization

- `.gitignore`: keep generated data, checkpoints, temporary files, and run artifacts out of version control.
- `mathematical_overview.ipynb`: mathematical definitions and motivation.
- `mnist_experiment`: the primary controlled continual-learning experiment. Follow its local `AGENTS.md`.
- `demo`: applied robotics demonstrations based on findings that survive the controlled experiments.
- `src`: coherent shared implementations used by experiments and demonstrations.
- `cache`: datasets, immutable run artifacts, large reference matrices, and model checkpoints.
- `test/unit`: fast deterministic tests for estimators, matrix operations, configuration logic, and artifact handling.
- `test/integration`: slower end-to-end smoke runs. Run these only when requested or when explicitly validating the experiment pipeline.

The previous synthetic numerical experiment has been retired because its Gaussian auxiliary-score assumptions conflict with the corrected LFU mathematics. Do not recreate that design without a new specification.

## Engineering requirements

- Prefer existing project patterns and PyTorch functionality over new abstractions.
- Keep mathematical conventions explicit in code. In particular, document whether a formula uses scores $s$ or loss gradients $g=-s$.
- Compute per-sample gradients where required. A batch-mean outer product is not a mean of per-sample outer products.
- Use Hessian-vector products; never materialize a full per-sample Hessian.
- Symmetrize numerical Fisher updates before spectral operations.
- Record diagnostics before PSD projection so projection cannot hide estimator instability.
- Keep representations interchangeable behind a small interface, but do not force dense, diagonal, and low-rank-plus-diagonal methods into an abstraction that obscures their numerical differences.
- Add succinct comments only where the mathematical intent is not evident from the code.

## Testing requirements

Add fast tests whenever a feature or bug fix is introduced. At minimum, the experiment implementation should eventually cover:

- score/loss-gradient sign conventions;
- the $UBU^T$ LFU factorization;
- HVPs against finite differences;
- lagged-direction scheduling;
- dense and diagonal PSD projection;
- low-rank-plus-diagonal matrix-vector products;
- deterministic paired configuration generation;
- immutable-run collision and completion behavior;
- a tiny CPU smoke trajectory.

Run unit tests often. Do not run long experimental integrations or robotics tests without the user's request.
