
# Amari-Chenstov Updates 

Elastic Weight Consolidation (EWC) and Low Rank Adapters (LoRA) technology [1] show great potential for tuning AIs toward specialization, 
which would be useful in personalization, remote autonomy, and reducing overall compute burdens while updating models. 
These methods ultimately work by estimating the model's Fisher Information Matrix (FIM), 
which is often done in a continual or moving average fashion. 
A moving average has a solid chance to remain sufficiently up-to-date. 
However, this work explores whether or not additional first-order Taylor series adjustments can help keep the FIM estimate more-accurate. 
Provided updates occur in batches of data, we may invoke the Central Limit Theorem (CLT) to enjoy approximately Gaussian-distributed steps over the statistical manifold. 
In this case, the first-order Taylor Series adjustment is done by estimating the Amari-Chentsov tensor [2] or its derivatives. 

## Math sketch 

EWC is applied to a continual learning context, where data are obtained form a series of independent but not necessarily identically distributed observations $X_i \sim f_X(x; \theta_i)$. 
Again, assuming each $\theta_i$ enjoys an $iid$ batch of observations, we may invoke CLT and get the following Taylor Series-based approximation.

$$ \left[ \mathcal I (\theta + d\theta) \right]_{ij} = \mathbb E_{\theta + d\theta} \partial_i \ell \partial_j \ell \approx \mathbb E_\theta \partial_i \ell \partial_j \ell + \mathbb E_\theta \partial_i \ell \partial_j \ell \partial_k \ell d\theta^k = \left[ \mathcal I (\theta) + C : d\theta \right]_{ij} $$

where 
- $\partial_i := \partial / (\partial \theta_i)$, 
- $\ell := \log f_X(X; \theta)$, 
- $C_{ijk} := \mathbb E_\theta \partial_i \ell \partial_j \ell \partial_k \ell$ (the Amari-Chentsov tensor), 
- $a_i b^i := \sum_i a_i b_i$ (Einstein summation notation), and 
- $C : v := C_{ijk} v^k $. 

## Experimental considerations 

In contexts where $\theta_t$ moves incredibly slowly per many observations gathered, 
the math is clear: the Amari-Chentsov tensor will produce more-accurate FIM estimates. 
Of course, with even greater data volumes per $\theta_t$ step, 
we could just entirely re-estimate the FIM entirely and not need Amari-Chentsov updates at all. 
As the third term in the log likelihood's Taylor Series expansion, 
the Amari-Chentsov is the expectaion of triple outer-product of score vectors, 
a high-variance entity. 
Further, as a rank-3 tensor a naive representation will take $O(p^3)$ space per $p$ tunable model parameters, absolutely massive. 
So, there are statistical and numerical considerations the ultimately challenge the practical effectiveness of this technique. 
Hence, an applied experiment is begged. 

We will run an MNIST experiment with small models studying:
1. When do Amari-Chentsov updates add value beyond FIM moving average estimates? 
2. Which strategies are useful in controlling the high-variance of $\widehat{C : d\theta}$? 

We'll then run another MNIST experiment with 
(1) scalable low-rank approximations and 
(2) small batch sizes (as low as 1),
and attempt to reproduce results results observed in our first round of MNIST experiments. 
If successful, this'll give us a foundation for applied success, because:
1. low-rank approximations of the FIM and Amari-Chentsov tensor keep space complexity managably bounded for large models, and 
2. single observation batch updates drastically reduce VRAM consumed by activations while estimating the FIM and Amari-Chentsov tensor.   

We'll then conclude with a robotics demonstration illustrating how this technology can fine-tune a Visual Language Model (VLM) for specific application. 

## Mathematical Model

For every $\theta \in \Theta$ assume the existence of a pairing $d(\theta) \in \Theta$. 
Whenever a model is at $\theta$, it draws samples 
from $d(\theta)$. 
When we update our model with the new data to some $\widehat{d(\theta)}$, 
then we say the model is now at $\theta \gets \widehat{d(\theta)}$. 
Naturally, much of our work assumes $d\theta := d(\theta)$ is a small value. 

It is convenient to further assume the existence of a well-defined and finite gradient field $\mathbb E_\theta \nabla_\theta \ell$ 
and Hessians $\mathbb E_\theta \nabla_\theta^2 \ell$, Amari-Chentsov tensors $\mathbb E_\theta \nabla_\theta^3 \ell$, 
and fourth derivatives. 
These assumptions allow us to construct a coherent theory of Amari-Chenstov-updated EWC regularizers. 
If we further assume
(1) sufficiently many samples per $\theta_t$ to invoke CLT and thus Gaussian steps, and 
(2) choose $d(\theta)$ as a correct function of the gradient field $\mathbb E_\theta \nabla_\theta \ell$, 
then we enjoy the existence of a Stochastic Differential Equation (SDE) well-approximating our process over the statistical manifold. 
 
To derive the EWC regularizer under a frequentist paradigm, 
we introduce a mixture model according to _observed_ Bernoulli variable $M_t$, 
such that:
- $\mathbb P[ M_t = 1] = \pi = 1 - \mathbb P[M_t = 0]$, 
- $(X_t \, | \, M_t = 0) \sim f_X(x; \theta_t)$, and
- $(X_t \, | \, M_t = 1) \sim f_X(x; \theta_t + d\theta_t) $.

To reproduce the EWC regularizer, 
break our observation vector into two sub-vectors $X = (X^t, X^{t+1})$, where 
- $X_i^t = (X_i \, | \, M_i = 0)$, and
- $X_i^{t+1} = (X_i \, | \, M_i = 1)$. 

Now break the log likelihood into two parts.

$$ \log f_X(X;\theta + d\theta) = \log f_X(X^{t+1}; \theta + d\theta) + \log f_X(X^t ; \theta + d\theta) $$

$$ \approx \log f_X(X^{t+1}; \theta + d\theta) + \log f_X(X^t ; \theta) + d\theta^T \nabla_\theta \log f_X(X^t; \theta) + 2^{-1} d\theta \left(\nabla_\theta^2 \log f_X(X^t ; \theta)\right) d\theta $$

$$ \approx_{a.s.} \log f_X(X^{t+1}; \theta + d\theta) + 0 - 2^{-1} \left( \sum_{i=1}^n \mathbb I_{\{M_i = 0\}} \right) d\theta^T \mathcal I (\theta) d\theta $$

We've almost recovered EWC, 
but the regularizer constant takes an elegant $(1 - \pi)/2$ form under MLE optimization. 

$$ \hat \theta_{t+1} = \arg\max_{d\theta} n^{-1} \log f_X(X; \theta + d\theta) \approx n^{-1} \log f_X(X^{t+1}; \theta + d\theta) - \frac{1-\pi}{2} d\theta^T \mathcal I (\theta) d\theta $$

This has approximate asymptotic distribution $\hat \theta_{t+1} \sim \mathcal N \left( \pi d\theta_t + \theta_t, \; \pi \mathcal I^{-1}(\theta_t+d\theta_t) / n \right)$.

### SDE construction 

For each integer $k \geq 0$, there exists triangular array $\{ X_{i,n}^k \}_{i=1}^n$ of observations with known $M_{i,n}^k$ mixture variables. 
Assume the existence of a single gradient field $b(\theta)$ on $\Theta$, 
allowing us to coherently define stochastical integral paths over the statistical manifold. 
This gives us a clean asymptotic approximation:

$$ \theta_{k+1,n} \approx \theta_{k,n} + \pi b(\theta_{k,n})/n + \sqrt{\pi \mathcal I^{-1} (\theta_{k,n}) /n } \, \xi_k, \; \xi_k \sim_{iid} \mathcal N(0, \; I_p)$$

Scaling with $k = \lfloor nt \rfloor$ and $n \to \infty$ produces the SDE.

$$ d\Theta_t = \pi b(\Theta_t) dt + \sqrt{\pi \mathcal I^{-1} (\Theta_t)} dW_t $$

### Optimal $\pi$ as stochastic control 

TODO: ill-posed, do not vary $\| d\theta_t \|$. Control $\pi$.

Unproven here, an MSE-optimal estimate of $\theta_{t+1}$ is obtained by choosing $\pi = \mathrm{tr}\left[ \mathcal I^{-1}(\theta_t) n^{-1} \right] / \left( 2 \| d\theta_t \|^2 \right)$. 
Choosing $\pi = 1/2$ constantly, we get $n = \mathrm{tr}\left[ \mathcal I^{-1}(\theta_t) \right] / \| d\theta_t \|^2 $.
However, this implies $\| d\theta_t \| = \sqrt{\mathrm{tr}\left[ \mathcal I^{-1}(\theta_t) \right] / n} = O(n^{-1/2})$, 
a contradiction. 

So, we'll instead choose a different limit and thus limiting process. 
A model at $\theta_{k,n}^*$ has true generating point $ \theta_{k,n}^* + d\theta_{k,n}^* = \theta_{k,n}^* + b(\theta_{k,n}^*) / \sqrt{n} $. 
This gives us update equation:

$$ \theta_{k+1,n}^* = \theta_{k,n}^* + \pi \frac{b(\theta_{k,n}^*)}{\sqrt n} + \sqrt{\pi \mathcal I^{-1}(\theta_{k,n}^*) / n} \, \xi_k $$

Taking $\pi = 1/2, k = \lfloor \sqrt n t \rfloor$, and assuming $b$ and $\mathcal I^{-1}$ are Lipschitz continuous, 
we get a new discrete process without a contradiction.
- $\Delta_{k,n} := \theta_{k+1,n}^* - \theta_{k,n}^*$
- $ \Rightarrow \mathbb E [ \Delta_{k,n} \, | \, \mathcal F_k ] = 2^{-1} b(\theta_{k,n}^*) / \sqrt n $ and
- $ \mathrm{Cov} [ \Delta_{k,n} \, | \, \mathcal F_k ] = 2^{-1} \mathcal I^{-1}(\theta_{k,n}^*) / n $. 
- $ \sum_{j=1}^{k-1} \mathbb E [ \Delta_{j,n} \, | \, \mathcal F_j ] = \sum_{j=1}^{\lfloor \sqrt n t \rfloor-1} b(\theta_{j,n}^*) / (2 \sqrt n) = O( \sqrt n / \sqrt n) $ by Lipchitz, so the deterministic term converges.
- $ \sum_{j=1}^{k-1} \mathrm{Cov} [ \Delta_{j,n} \, | \, \mathcal F_j ] = \sum_{j=1}^{\lfloor \sqrt n t \rfloor-1} 2^{-1} \mathcal I^{-1}(\theta_{j,n}^*) / n = O( \sqrt n / n)$ by Lipschitz, so the stochastic term vanishes.

So, optimal control causes randomness to vanish with large sample sizes, 
leaving us with a path integral $\Theta_t^* = \Theta_0^* + 2^{-1} \int_0^t b(\Theta_s) ds$ 
or simply $\dot \Theta_t^* = 2^{-1} b( \Theta_t^*)$. 
Of course, no true sample size is ever infinite, 
so we may find it pragmatic to approximately model the discrete process with small-noise SDE $\Theta_t^\varepsilon$:

$$ d\Theta_t^\varepsilon = 2^{-1} b(\Theta_t^\varepsilon) dt + \sqrt{\varepsilon / 2} \mathcal I^{-1/2}(\Theta_t^\varepsilon) dWt, \; \varepsilon = n^{-1/2}.$$

### Why MNIST is mathematically comparable to Reinforcement Learning (RL) 

In RL, the agent at $\theta_t$ can be imagined to sample data from some unknown $\theta_{t+1}$. 
Upon a model update, the model _moves_ to $\theta_{t+1} \gets \hat \theta_{t+1}$. 
So, our model reduces RL to estimating a slowly moving target over a set of distributions. 
This, of course, invites distortions to our FIM estimate, so perhaps motivates Amari-Chentsov updates. 

Our MNIST experiment will initially fit a model to digits 0 through 8, inclusive. 
This initial model is fit on a large amount of data, 
so represents the EMC + LoRA context: fine tuning an initial large model. 
We will then start sampling 9s with some probability $p$, starting of course with $p = 0$. 
Then we will slowly increase $p \to 1/2$, until 9s compose about 50% of observed samples. 
Our primary metric: accuracy in classifying 9s. 

Both RL and our MNIST experiment, under our model, 
are estimating a slowly moving target $\theta_i$ from independent samples $X_i \sim f_X(x; \theta_i)$. 
In this sense, they are comparable. 

## Citations 

[1] Y. Zheng, Y. Zhang, J. van de Weijer, G. M. van de Ven, S. Du, X. Zhang, and Z. Tian, [*Revisiting Weight Regularization for Low-Rank Continual Learning*](https://arxiv.org/abs/2602.17559), arXiv:2602.17559, 2026.

[2] S. Amari and H. Nagaoka, *Methods of Information Geometry*, American Mathematical Society, 2000. 
