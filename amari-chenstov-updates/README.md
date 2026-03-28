
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

$$ \left[ \mathcal I (\theta + d\theta) \right]_{ij} = \mathbb E_{\theta + d\theta} \partial_i \ell \partial_j \ell \approx = \mathbb E_\theta \partial_i \ell \partial_j \ell + \mathbb E_\theta \partial_i \ell \partial_j \ell \partial_k \ell d\theta^k = \left[ C : d\theta \right]_{ij} $$

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
Whenever a model is at $\theta$, it draws samples from $d(\theta)$. 
When we update our model with the new data to some $\widehat{d(\theta)}$, 
the we say the model is now at $\theta \gets \widehat{d(\theta)}$. 
Naturally, much of our work assumes $d\theta := d(\theta)$ is a small value. 

It is convenient to further assume the existence of a well-defined and finite gradient field $\mathbb E_\theta \nabla_\theta \ell$ 
and Hessians $\mathbb E_\theta \nabla_\theta^2 \ell$, Amari-Chentsov tensors $\mathbb E_\theta \nabla_\theta^3 \ell$, 
and fourth derivatives. 
These assumptions allow us to construct a coherent theory of Amari-Chenstov-updated EWC regularizers. 
If we further assume
(1) sufficiently many samples per $\theta_t$ to invoke CLT and thus Gaussian steps, and 
(2) choose $d(\theta)$ as a correct function of the gradient field $\mathbb E_\theta \nabla_\theta \ell$, 
then we enjoy the existence of an Stochastic Differential Equation (SDE) well-approximating our process over the statistical manifold. 

### Why MNIST is mathematically comparable to Reinforcement Learning (RL) 

In RL, the agent at $\theta_t$ can be imagined to sample data from some unknown $\theta_{t+1}$. 
Upon a model update, the model _moves_ to $\theta_{t+1} \gets \hat \theta_{t+1}$. 
So, our model reduces RL to estimating a slowly moving target over a set of distributions. 
This, of course, invites distortions to our FIM estimate, so perhaps motivates Amari-Chentsov updates. 

Our MNIST experiment will initially fit a model to digits 0 through 8, inclusive. 
This initial model is fit on a large amount of data, 
so represents the EMC + LoRA context: fine tuning an initial large model. 
We will then start sampling 9s with some probability $p$, starting of course with $p = 0$. 
Then we will slowly increase $p \to 1$, until 9s compose about 10% of observed samples. 
Also, as $p$ increases, we'll adjust the loss function to emphasize correct classification of 9s. 

Both RL and our MNIST experiment, under our model, 
are estimating a slowly moving target $\theta_i$ from independent samples $X_i \sim f_X(x; \theta_i)$. 
In this sense, they are comparable. 

## Citations 

[1] Y. Zheng, Y. Zhang, J. van de Weijer, G. M. van de Ven, S. Du, X. Zhang, and Z. Tian, [*Revisiting Weight Regularization for Low-Rank Continual Learning*](https://arxiv.org/abs/2602.17559), arXiv:2602.17559, 2026.

[2] S. Amari and H. Nagaoka, *Methods of Information Geometry*, American Mathematical Society, 2000. 
