# Instructions for coding agents 

You are helping to implement an AI & robotics experiment. 
Most instructions are _not_ listed here, but are referenced from this document. 
Given the scientific nature of this project, there is no clean collection of user stories for you; 
this is not a waterfall-style project. 
Instead, overall intention is stated at outset and the project proceeds in agile fashion. 
Expect conversational & collaborative exploration with the experimenter and occasional changes of strategy. 

**Rules:**
1. Never `commit`, `tag`, or `push` with `git`. That is the experimenter's responsibility. 
2. Maintain a clean repo with a single `/.gitignore` file. 
3. Maintain a `pytest` unit testing suite. The overall test suite should be comprehensive, ensuring new development doesn't break existing features unintentionally. If accomplishing this requires the test suite to take a few minutes to run, that's fine. 
4. Run your unit testing suite frequently. 
5. Develop a few integration tests in collaboration with the experimenter. Since robots are involved, integration tests are both important but also hard to execute. Propose them when perhaps important, but expect only a few to be implemented. 
6. Use `pytest` markers to distinguish default tests from unusual testing requirements. Expected markers: `slow`, `gpu`, `integration`, and `robot`. 
7. We'll structure this repo like a Python package-building project but also with data science elements. Here's a minimum list of directories & files:
   1. `/src/` stores all software assets. 
   2. `/tests/unit` stores all unit tests, implemented in `pytest`. 
   3. `/tests/integration` stores all integration tests, implemented in `pytest`. 
   4. `/build/` is an ephemeral directory used to store intermediary build artifacts, `git`-ignored, and never committed. 
   5. `/dist/` is an ephemeral directory used to store build outputs like `.whl` files, `git`-ignored, and never committed. 
   6. `/artifacts/models/` is an ephemeral directory used to store deep learning files, `git`-ignored, and never committed.
   7. `/artifacts/data/` is an ephemeral directory used to store sampled experimental data, `git`-ignored, and never committed.
   8. `/artifacts/manifests/ephemeral` is an ephemeral directory used to store machine-specific references, `git`-ignored, and never committed. For example, local IP addresses could be stored here. 
   9. `/artifacts/manifests/tracked` is a non-ephemeral directory used to store references. Content here is not necessarily user-facing, but must be safe to version. Store changing project defaults here, like VLM specifications. Experiment-specific configs may override these defaults. 
   10. `/config/` stores experimenter-facing configuration files used to control how _all_ experiments are executed. Keep experimental variables (like hyperparameters) out of here, since they'll be varied within experiments.
   11. `/notebooks/` stores conceptual documents, like math. 
   12. `/experiments/configs/` stores experiment-specific default configs. Defaults may be overridden in notebooks after loading. 
   13. `/experiments/runs/` stores outputs per experimental runs. Keep each run under 1MB, so we can upload to GitHub and not slow `git` management. Use this data to memoize and avoid re-running unnecessary experimental compute. Raw observations belong in `/artifacts/data/`; distilled results belong here. 
   14. `/experiments/reports/` stores notebooks for executing and presenting experimental results. 
   15. `/docs/agents/` stores additional documentation for you. 
   16. `/docs/humans/` stores documentation for human repo readers. It should describe essential mechanics like how to build, setup, & execute experiments. 
   17. `/scripts/` stores CICD-like Python scripts facilitating builds, tests, & installs. 
   18. `/pyproject.toml` describes the package built for running the robot. We won't have other packages in this project. 
   19. `/requirements-server.txt` describes all requirements needed to run experiments and run the GPU-capable server manipulating the robot. Heavier installs are allowed here. 
   20. `/requirements-robot.txt` describes all requirements that need to run the robot. The robot's onboard computer is just a Raspberry Pi, so this install must be light. The robot's Raspberry Pi hosts a minimal `Flask` server, while all substantial AI computation is run on the GPU-capable server. 
8. No large data files are to ever be `git`-committed. For example, deep learning models and experimental sampling data should not be committed. Use the `/.gitignore` file to manage this. 
9. Maintain a coherent, elegant codebase, avoiding duplicative content. Propose refactors to the experimenter when repo elegance starts degrading. 

For running Python, you'll find a virtual environment at `~/.venv`.

**References:**
1. High-level experimental intention is described in the repo root's readme file, `/README.md`. 
2. Implementation notes are stored in `/docs/agents/`. This directory can store notes from you or guidance from the user. 
3. `/docs/agents/experimental-design-notes-1.md` are notes you authored after conversationally clarifying your understanding of `/README.md`. 
4. `/docs/agents/existing-artifacts.md` describes existing experimental artifacts I've authored which I'd like to integrate into this codebase and use as starter content. 
