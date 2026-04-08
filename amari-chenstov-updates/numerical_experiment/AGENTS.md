# Numerical experiment

According to [mathematical_overview](../mathematical_overview.ipynb), we should expect our RL programs to traverse their statistical manifolds like SDEs. 
However, because deep learning models are large, we'll only be able to track a small variety of sufficient statistics. 
Here, we'll see which combinations of FIM and Amari-Chentsov statistics produce practical results. 

Use these instructions to author `numerical_experiment/experiment.ipynb`. 
Do not add code to `src/`, since none of it will be resused in other experiments. 

## Random manifold generation 

Write a function `generate_manifold` that produces a deformation of $\mathbb{R}^{100}$, not a strict subset. 
The function does _not_ return a manifold parameterization, 
but instead returns the Reimann metric function `g_ij(theta)`. 
To construct `g_ij(theta)`, I recommend defining strong-convex potential 
`psi(theta) = 1/2 theta^T A theta + eps * sum_k a_k phi_k(theta)` with:
- `A` positive definite, to anchor global convexity. 
- `phi_k` smooth random features, to create interesting local geometry. 
- `eps` small enough that `nabla^2` psi stays positive definite everywhere. 

Convexity guarantees `g_ij(theta) = d_i d_j psi(theta)` is PD. 

The underlying manifold should have an interesting shape, 
something that'll create variation between experimental trials. 
That's where `phi_k(theta)` comes into play. 
I'd like you to randomly add these geometric features:
- low-frequency sinusoidal features along random directions `phi_k(theta) = cos(w_k^T theta + b_k)`. 
- smooth localized bumps `phi_k(theta) = exp(-||P_k theta - c_k||^2 / (2 sigma_k^2))`. 
- ridge-like nonlinearities `phi_k(theta) = log cosh(w_k^T theta + b_k)`. 

Remember, 100 dimensions is a lot of space, so add a lots of these geometric features. 

I'll use this `g_ij(\theta)` metric to sample scores, simulating an RL agent-environment interaction. 

## Random gradient field generation 

Write a function `generate_gradient_field` which generates a vector field over the manifold produced by `generate_manifold`. 
While returning a gradient field, it is _not_ the gradient field of our generated manifold. 
It is meant to model the optimal next agent state given the agent's current state. 
This is part of what makes the experiment interesting, 
the interplay between two different potentials. 
Sometimes the agent's design makes it easy to learn the game. 
Other times, the game requires the agent to traverse difficult parts of its statistical manifold, 
directing it with this gradient field. 

The gradient field needs an attractor, but shouldn't be a naive parabaloid. 
I want interesting and diverse flows. 
To model this, I recommend definding a smooth scalar potential `V : M -> R` returning covector drift `X = -grad_g V`, 
given our Manifold's Riemann metric `g`. 
Further, let `V = V_anchor + eps * V_random`, 
where `V_anchor` has a guaranteed well, 
`V_random` introduces fun geometric shapes, 
and `eps` is small enough that `V` has guaranteed convexity. 

Construct similar to our manifold's potential: `V(x) = lambda * w(x; p) + eps * sum_k a_k phi_k(x)`. 
- `w(x; p)` is a smooth well centered at `p`, like squared geodesic distance near `p`
- `phi_k` are smooth basis functions on the manifold. 
- `a_k` are random coefficients. 
- `lambda` is large enough relative to `eps` that convexity is gauranteed. 

For `V`, I recommend using the same broad classes of `phi_k` as manifold generation, 
but with parameters chosen for interesting flow rather than metric stability. 
Because `V` only shapes the drift field, it can tolerate sharper and more dramatic features than `psi`. 

I recommend mixing these feature types:
- low-frequency sinusoidal features `phi_k(theta) = cos(w_k^T theta + b_k)`. 
  Use these to create broad valleys, large-scale directional biases, and long coherent excursions. 
  Sample `w_k` with relatively small norm so trajectories remain smooth over long distances. 
- smooth localized bumps `phi_k(theta) = exp(-||P_k theta - c_k||^2 / (2 sigma_k^2))`. 
  Use these to create local hills, side-wells, bottlenecks, and metastable regions. 
  Prefer medium and large `sigma_k`; very tiny bumps make the flow noisy rather than interesting. 
- ridge-like nonlinearities `phi_k(theta) = log cosh(w_k^T theta + b_k)`. 
  Use these to create elongated canyons and soft barriers that redirect trajectories without creating discontinuities. 

To keep `V` interesting, sample coefficients `a_k` with mixed signs, 
so `V_random` contains both hills and wells. 
Use many terms, but decay coefficient magnitude with feature sharpness, 
so high-frequency terms do not dominate the overall flow. 
I recommend using more bump and ridge terms than sinusoidal terms: 
sinusoids give the drift global structure, 
while bumps and ridges create the actual "adventure." 

As a starting point, try approximately 20% sinusoidal features, 50% bumps, and 30% ridges. 
Keep `eps` small relative to `lambda` so the anchor well remains the unique dominant attractor, 
while `V_random` still creates saddles, detours, and long transient excursions. 

Note that our well is not the origin, like in manifold generation. 
If they had the same attractor, the overall geometry would be too boring. 

The function should return `X(theta)`. 

## Sampling distribution 

There is no need to simulate any underlying distribution. 
Instead, we are statistically modelling our auxiliary score distributions, as described in [mathematical_overview](../mathematical_overview.ipynb). 
We'll _lean into_ our SDE results, and sample all scores from a multivariate Gaussian distribution. 
For a model with parameter `theta`, it will samples scores from `theta + X(theta)`. 
An easy way to do this is to sample from `\mathcal N ( 0, g_ij(theta + X(theta)))`. 

## Stochastic processes 

TODO optimal batches and small samples 
We'll model the $d\Theta_t$ and $d\Theta_t^*$ proceses as found in [mathematical_overview](../mathematical_overview.ipynb). 
Since this experiment studies the effects of different sufficient statistic estimation strategies, 
the FIM must be estimated from observed scores. 

Whenever scores are sampled, at least 2 updates occur:
1. The SDE location $\theta_t$ is updated via our regularized MLE equation $\theta_{t+1} = \arg\max_{d\theta} \log n_{t+1}^{-1} f_X(X; \theta_t + d\theta) - .5 (1-\pi) d\theta^T \widehat{\mathcal I} d\theta $, where $n_t$ is the total number of observations up to and including the $t^{th}$ batch. 
2. The FIM estimate is updated via $\widehat{\mathcal I} \gets \sum_{i=1}^{n_{t+1} - n_t} s_i s_i^T / n_{t+1} + \frac{n_t}{n_{t+1}} \widehat{\mathcal I} $. 
If the model tracks an Amari-Chenstov estimate $\hat C$, then the update is $\widehat{\mathcal I} \gets \Pi_{PSD} \left( \sum_{i=1}^{n_{t+1} - n_t} s_i s_i^T / n_{t+1} + \frac{n_t}{n_{t+1}} \widehat{\mathcal I} + C : d\theta \right) $, where $\Pi_{PSD}$ is a projection into the set of PSD matrices - I recommend using eigenvalue clipping. 
3. TODO strategies for $C$ and $C : d\theta$ estimation. TODO contend with `C = 0 if s_i ~ Gaussian :(` 

**Single sample process**: This models $d\Theta_t$, sampling one observation, then updating with a very small, constant $pi$ value. 
TODO

**Batched process**: TODO

## Experimental conditions 

TODO 

## Model initialization 

TODO initial FIM and Amari-Chentsov estimates 

## Metrics 

TODO 
