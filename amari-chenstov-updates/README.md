
# Amari-Chenstov Updates in Reinforcement Learning (RL) 

Elastic Weight Consolidation (EWC) and Low Rank Adapters (LoRA) technology [1] show great potential for tuning AIs toward specialization, 
which would be useful in personalization, remote autonomy, and reducing overall compute burdens while updating models. 
These methods ultimately work by estimating the model's Fisher Information Matrix (FIM), 
which is often done in a continual or moving average fashion. 
A moving average has a solid chance to remain sufficiently up-to-date. 
However, this work explores whether or not additional first-order Taylor series adjustments can help keep the FIM estimate more-accurate. 
Provided updates occur in batches of data, we may invoke the Central Limit Theorem (CLT) to enjoy approximately Gaussian-distributed steps over the statistical manifold. 
In this case, the first-order Taylor Series adjustment is done by estimating the Amari-Chentsov tensor [2] or its derivatives. 
See [mathematical_overview](mathematical_overview.ipynb) for more detail and a frequentist construction of EWC. 

## Why are experiments needed? 

Despite rigorous mathematical argument, this work explores the applied value of Amari-Chentsov updates. 
The big question: in applied contexts, when do Amari-Chentsov updates add value to RL? 
It is entirely possible a moving average Fisher Information Matrix (FIM) estimate is sufficient and fully effective. 

## Experimental design 

A series of experiments will be run, 
improving our understanding of Amari-Chentsov updates complicating science with applied rigor. 
1. **numerical_experiment**: is an abstract numerical experiment studying how tracking different sufficient statistics can most-optimally guide SDEs according to the theory derived in [mathematical_overview](mathematical_overview.ipynb). It'll help us understand numerical & statistical viability.
2. **mnist_experiment**: is an RL-equivalent, simple experiment helping us understand when Amari-Chentsov updates are most-useful. 
The primary task: correctly classify 9s as the sample proportion of 9s increases from 0% to 50%. 
To see how RL can be equivalent to an MNIST experiment, see [mathematical_overview](mathematical_overview.ipynb). 
Small models will be used. 
Approximations will be avoided when possible. 
Both online and batch fitting will be tested. 
3. **demo**: An RL and robotics demo will illustrate the applied benefits as discovered in experiments 1 and 2. 
Only online fitting will be illustrated because batch fitting user experience is poor - it's _no fun_ to wait for fitting. 

## Software architecture 

See [AGENTS.md](AGENTS.md) to understand how code and directories are organized. 

## Citations 

[1] Y. Zheng, Y. Zhang, J. van de Weijer, G. M. van de Ven, S. Du, X. Zhang, and Z. Tian, [*Revisiting Weight Regularization for Low-Rank Continual Learning*](https://arxiv.org/abs/2602.17559), arXiv:2602.17559, 2026.

[2] S. Amari and H. Nagaoka, *Methods of Information Geometry*, American Mathematical Society, 2000. 
