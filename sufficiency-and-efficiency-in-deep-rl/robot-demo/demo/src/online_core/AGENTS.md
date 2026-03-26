# Oneline Core Specification 

This document provides instruction to the agent on how to implement the online core in `src/online_core/online_ssr_agent.py` and its supporting modules. 

## Goal 

Implement a sub-abstraction layer inheriting from `SSRAgent` in `src/core/ssr_agent.py`.
The `oneline_core` is an `SSRAgent` variant with an alternative `fit` function, 
leveraging sufficient statistics to update the agent in an online fashion. 
This streamlines user experience & VRAM utilization. 
The `fit` function uses the SSR regularizer to perform an `optimizer.step()` per every single observation taken, immediately after it is observed. 
Then, because the gradient vector is already cached, 
it can also be immediately memorized into the Fisher Information Matrix (FIM) sufficient statistic. 
Actions are sent to the robot asynchronously as fitting and memorization occur. 
This avoids having to regenerate or store VLM activations (which are large) 

## Motivation 

I've been running experiments using a typical RL paradigm: collect data, optimize, repeat. 

## Mathematical argument 

For $X_{1:n} = (X_1, X_2, \ldots, X_n)$, 
observe that $\hat \theta_n = \arg\max_\theta n^{-1} \log f_X(X; \theta) $
$ = \arg\max_\theta n^{-1} \log f_X(X_n; \theta) + n^{-1} \log f_X(X_{1:n-1}; \theta) $ 
$ \approx_{a.s.} \arg\max_\theta n^{-1} f_X(X_n; \theta) + \frac{n-1}{2n} (\theta - \theta_{n-1})^T \mathcal I(\theta_{n-1}) (\theta - \theta_{n-1}) $ 
assuming $X_i \sim_{iid} f_X(x; \theta_{n-1}) $ and $n-1$ large. 
So, for each new observation, we'll use estimator $\hat \theta_n = \arg_max_\theta n^{-1} \log f_X(X_n; \theta) - \frac{n-1}{2n}(\theta - \hat \theta_{n-1})^T \hat{\mathcal I}(\theta_{n-1}) (\theta - \hat \theta_{n-1}) $.
Some practical accommodations:
1. $n$ may be given a max value to ensure learning doesn't slow too much.
2. If initial information in $\hat{\mathcal I}(\theta_0)$ is very valuable, it may be up-scaled to simulated to protect it.

To maintain an up-to-date estimate for $\hat{\mathcal I}(\theta_{n-1})$ can be done rigorously, 
but requires some numerical and statistical heuristics. 
In this work, we assume the initial model at $\theta_0$ was fit with a large dataset, 
so both $\hat \theta_0$ and $\hat{\mathcal I}(\theta_0)$ are accurate. 
Using a Talor series expansion, we see that 
$\left[ \mathcal I(\theta + d\theta) \right]_{ij} = \mathbb E_{\theta + d\theta} \partial_i \ell \partial_j \ell $
$ \approx \mathbb E_\theta \partial_i \ell \partial_j \ell + \mathbb E_\theta \partial_i \ell \partial_j \ell \partial_k \ell d \theta^k =: \left[ \mathcal I(\theta) + C : d\theta \right]_{ij} $, 
assuming normal increments, $\partial_i := \partial/\partial \theta_i$, and $\ell := \log f_X(X_{1:n-1}; \theta) $.
Notice that $C$ is the Amari-Chentsov tensor.
Practical accommodations:
1. We're assuming Gaussian increments. This is argued via locally via Local Asymptotic Normality (LAN), but not globally.
2. $C$ is a rank-3 tensor of likely unmanagable size, so we won't estimate it directly. Instead, we'll estimate $C : d\theta$.
3. $\widehat{C : d\theta}$ is a high-variance estimator. We'll use a moving average over observations and shrink its effect when variance is large.
4. This update only works if our initial $\mathcal I(\theta_0)$ estimate is accurate, so we assume the initial model is fit on a very large dataset. In application, models don't ship with an $\mathcal I(\theta_0)$ estimate, so we'll have to manually sample our own data from $\theta_0$ as an initialization step.

To argue Gaussian increments with an asymptotic approximation, 
it's not enough to have a single observation $X_n \sim f_X(x; \theta_n)$, 
otherwise its effect will average-out to zero. 
Instead, we argue a proportion $\pi$ of the sample is distributed according to $\theta_n$. 
To achieve this, we assume a mixture model where $(X_i | M_i) \sim_{iid} f_X(x; \theta_n)^{M_i} f_X(x; \theta_{n-1})^{1-M_i}$,
 $M_i \in \{0,1\}$, and $\mathbb P [M_i = 1] = \pi$. 
In RL experiments, we know which observations come before and after a model update, so each $M_i$ is observed. 

To argue Gaussian increments, we assume a consistently large batch of new observations. 
Since we use an asymptotic approximation, any finite batch size will average to zero, 
so instead we insist on a proportion $\pi$ of samples that are new. 
Then, in application, we'll drive this proportion down to a single observation, 
which will degrade normality (safely) and give us a substantial VRAM savings.
1. Assume a mixture model depending on Bernoulli $M_i$ with $\mathbb P [M_i = 1] = \pi$. 
$X_i$ observations with $M_i = 0$ from before a model update so $(X_i | M_i = 0) \sim_{iid} f_X(x; \theta)$, 
while $M_i = 1$ are from after so $(X_i | M_i = 1) \sim_{iid} f_X(x; \theta + d\theta)$
Since we control when model updates occur, all $M_i$ values are observed and known. 
$\pi > 0$ is constant, so our asmyptotic distribution will have a non-zero effect from new data. 
The estimate is asymptotically $\hat \theta_0 = \arg\max_{d\theta} \log f_X(X; \theta+d\theta) - (1-\pi)/2 d\theta^T \mathcal I d \theta$, so has an approximately Gaussian distribution. 
2. In peronalized AI, VRAM is precious, 
so we explore the effect of taking $\pi$ so low that our obseved estimate has only 1 sample. 
It is _not_ Gaussian distributed, but does enjoy some degree of approximate safety, as described below.

Here, we argue that a slightly elevated $\pi$ value causes single-observation updates to be safe and productive, 
under mild regularity conditions.
Assume $\log f_X(X_n; \theta)$ is twice continuously differentiable, 
that $\nabla \log f_X := \nabla_\theta \log f_X(X_n; \theta)$ is Lipschitz continuous in $\theta$, 
and existence of $\mathbb E \nabla_\theta^2 \log f_X$. 
Then our estimate is of the form $\widehat{d\theta} = \arg\max_{d\theta} \log f_X(X_n; \theta+d\theta) - 2^{-1}\lambda d\theta^T \mathcal I d\theta \approx \arg\max_{d\theta} d\theta^T \nabla \log f_X + 2^{-1} d\theta^T \nabla^2 \log f_X d\theta - 2^{-1}\lambda d\theta^T \mathcal I d\theta $. 
WLOG, we may take $\mathbb{E} \nabla \log f_X = 0$ by choosing $\lambda$ sufficiently large to dominate its Lipschitz continuity. 
Further WLOG, we take $\nabla^2 \log f_X := \nabla_\theta^2 \log f_X(X_n; \theta) \leq 0$ again by choosing sufficiently large $\lambda$ to dominate it. (TODO check thin tail assumption) 
So, the program's solution is $0 = \nabla \log f_X + \nabla^2 \log f_X \widehat{d\theta} - \lambda \mathcal I \widehat{d\theta} $
$ = \nabla \log f_X - (\nabla^2 \log f_X - \lambda \mathcal I) \widehat{d\theta} $
$ \Rightarrow \widehat{d\theta}^T(-\nabla^2 \log f_X + \lambda \mathcal I)\widehat{d\theta} = \widehat{d\theta}^T \nabla \log f_X $ 
$ \Rightarrow \lambda \| \widehat{d\theta} \|_{\mathcal I}^2 \leq \widehat{d\theta}^T(-\nabla^2 \log f_X + \lambda \mathcal I)\widehat{d\theta} = \widehat{d\theta}^T \nabla \log f_X = \widehat{d\theta}^T \mathcal I^{1/2} \mathcal I^{-1/2} \nabla \log f_X = (\mathcal I^{1/2} \widehat{d\theta})^T (\mathcal I^{-1/2} \nabla \log f_X) \leq \| \mathcal I^{1/2} \widehat{d\theta} \|_2 \| \mathcal I^{-1/2} \nabla \log f_X \|_2 = \| \widehat{d\theta} \|_{\mathcal I} \| \nabla \log f_X \|_{\mathcal I^{-1}} $
$ \Rightarrow \| \widehat{d\theta} \|_{\mathcal I} \leq \lambda^{-1} \| \nabla \log f_X \|_{\mathcal I^{-1}} $. 
So, $\widehat{d\theta}$ in information distance is bounded by the $\lambda^{-1}$ scaled score in covariance distance. 
So, it won't do much harm relative to larger sample. 
However, if $\lambda$ isn't tight, then we run the risk if not getting enough information from new data. 
That's why it's important to construct our estimation paradigm rigorously and choose $\lambda$ with a wealth of theoretical context. 

## Constraints

1. Do not specify a `loss` function in `OnlineSSRAgent` because it is still an abstract class. 

## Implementation tasks 

1. **Lanczos variant**: Implement a variant of `multiply_fisher_information` as found in `src/core/lanczos.py`
2. 