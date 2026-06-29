# Instructions for coding agents 

You are helping to implement an AI & robotics experiment. 
Most instructions are _not_ listed here, but are referenced from this document. 
Given the scientific nature of this project, there is no clean collection of user stories for you; 
this is not a waterfall-style project. 
Instead, overall intention is stated at outset and the project proceeds in agile fashion. 
Expect conversational & collaborative exploration with the experimenter and occaisional changes of strategy. 

**Rules:**
1. Never `commit`, `tag`, or `push` with `git`. That is the experimenter's responsiblity. 
2. Maintain a clean repo with a single `/.gitignore` file. 
3. Maintain a `pytest` unit testing suite. The overall test suite should be comprehensive, ensuring new development doesn't break existing features unintentionally. If accomplishing this requires the test suite to take a few minutes to run, that's fine. 
4. Run your unit testing suite frequently. 
5. Develop a few integration tests in collaboration with the experimenter. Since robots are involved, integration tests are both important but also hard to execute. Propose them when perhaps important, but expect only a few to be implemented. 
6. We'll structure this repo like a Python package-building project but also with data science elements. Here's a minimum list of directories & files:
   1. `/src/` stores all software assets. 
   2. `/test/unit` stores all unit tests, implemented in `pytest`. 
   3. `/test/integration` stores all integration tests, implemented in `pytest`. 
   4. `/artifacts/build/` is an emphemeral directory used to store intermediary build artifacts, `git`-ignored, and never committed. 
   5. `/artifacts/dist/` is an emphemeral directory used to store build outputs like `.whl` files, `git`-ignored, and never committed. 
   6. `/artifacts/models/` is an emphemeral directory used to store deep learning files, `git`-ignored, and never committed.
   7. `/artifacts/data/` is an emphemeral directory used to store sampled experimental data, `git`-ignored, and never committed.
   8. `/artifacts/manifests/ephemeral` is an emphemeral directory used to store machine-specific references, `git`-ignored, and never committed. For example, local IP addresses could be stored here. 
   8. `/artifacts/manifests/tracked` is a non-emphemeral directory used to store references. For example, VLM specifications and unexposed configs go here. 
   9. `/config/` stores experimenter-facing configuration files used to control how _all_ experiments are executed. Keep experimental variables (like hyperparameters) out of here, since they'll be varied within experiments.
   10. `/notebooks/` stores conceptual documents, like math. 
   11. `/experiments/configs/` stores experiment-specific default configs. Defaults may be overriden in notebooks after loading. 
   12. `/experiments/runs/` stores outputs per experimental runs. Try to keep data small enough that we can upload to GitHub and not slow `git` management. Use this data to memoize and avoid re-running unnecessary experimental compute. 
   13. `/experiments/reports/` stores notebooks for executing and presenting experimental results. 
   14. `/docs/agents/` stores additional documentation for you. 
   15. `/docs/humans/` stores documentation for human repo readers. It should describe essential mechanics like how to build, setup, & execute experiments. 
   16. `/scripts/` stores CICD-like Python scripts facilitating builds, tests, & installs. 
   17. `/pyproject.toml` describes the package built for running the robot. We won't have other packages in this project. 
   18. `/requirements-server.txt` describes all requirements needed to run experiments and run the GPU-capable server manipulating the robot. Heavier installs are allowed here. 
   19. `/requirements-robot.txt` describes all requirements that need to run the robot. The robot's onboard computer is just a Raspberry Pi, so this install must be light. The robot's Raspberry Pi hosts a minimal `Flask` server, while all substantial AI computation is run on the GPU-capable server. 
7. No large data files are to ever be `git`-committed. For example, deep learning models and experimental sampling data should not be committed. Use the `/.gitignore` file to manage this. 
8. Maintain a coherent, elegant codebase, avoiding duplicative content. Propose refactors to the experimenter when repo elegance starts degrading. 

**References:**
1. High-level experimental intention is described in the repo root's readme file, `/README.md`. 
2. Implementation notes are stored in `/docs/agents/`. This directory can store notes from you or guidance from the user. 
3. `/docs/agents/experimental-design-notes-1.md` are notes you authored after conversationally clarifying your understanding of `/README.md`. 
