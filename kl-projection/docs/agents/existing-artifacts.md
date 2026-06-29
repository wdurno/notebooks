# Existing artifacts 

This project leverages existing artifacts from other experiments. 
We will integrate these assets into a single codebase, improving organization and codebase coherency. 
To avoid machine-specific absolute paths, paths will be given relative to this repo's root. 
For example, `../kl-projection` is a self-reference. 

Artifacts for consideration:
1. Math at `../amari-chenstov-updates/mathematical_overview.ipynb` generalizes the REINFORCE algorithm to extend Frequentism through Information Geometric mechanics. The experiment concluded when a numerically intractible blocker was discovered. However, it coherently defines an online learning mechanism through single observation batches. Conveniently, it also has an optimal learning result, which we may leverage. Please store the result under `/notebooks/` and leverage its _single observation batches_ mechanic in this experiment's _phase 3_. 
2. Server code draft at `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/` illustrates a first attempt at implementing this experiment, albeit without KL-projection initialization. It was during this work that I discovered the need for a two-speed model hierarchy and made an attempt at online learning, so we'll need to refactor the codebase as we bring into our new paradigm. This take time & collaboration between us. Some things I'd like to retain:
   1. `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/data/` stores existing sampled experimental data. I'd like to re-use it and maintain format, if possible. Robotics data is valuable and should be reused when possible. 
   2. The demo includes speech-to-text and text-to-speech capabilities, facilitating verbal communication between the experimenter and the robot. It's a wonderful feature I'd like to retain. 
   3. EWC is conceptually extended to low-rank approximations through a $LL^T + \Lambda, L \in \mathbb{R}^{p \times r}, p > r$ representation. The numerical key to this is leveraging the `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/src/core/lanczos.py` package. This'll allow me to experiment with elliptical FIM approximations in this experiment's _phase 3_. 
   4. The integration test suite is particularly useful. 
3. `../../picar-v-rl-env/` encodes software for building and running the Raspberry Pi-based `Flask` server on the robot. I want it integrated into this codebase so I can version the whole experiment together, not across separate repos. 

Do not modify any of these referenced artifacts. 
Only read or copy them. 
Modifications will be made within _this_ repo. 