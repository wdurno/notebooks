# LoRA Fisher Eigenvalue Distribution Experiment

This project studies how LoRA fine tuning changes the empirical Fisher
information spectrum in a small continual-learning MNIST setting.

The motivating question is whether the Fisher information matrix estimated in a
LoRA parameter subspace is less severely skewed than the Fisher information
matrix estimated over the full neural network parameters. Karakida, Akaho, and
Amari show that, under mean-field assumptions for deep networks, most Fisher
eigenvalues concentrate near zero while the maximum eigenvalue can become very
large. That predicted heavy skew is statistically and numerically challenging
for methods that require both local Fisher estimates and useful inverse-Fisher
approximations, so this experiment asks whether LoRA's restricted fine-tuning
geometry helps avoid it.

## Experiment Summary

The task is constructed from MNIST as a controlled fine-tuning problem. A base
classifier is first trained on digits 0 through 8 only, while retaining a
10-class output head. Digit 9 is withheld during this initial phase.

After the base model converges, LoRA adapters are enabled on all linear layers
and the base weights are frozen. The sampling distribution is then shifted over
time using a parameter `t` in `[0, 1]`:

- with probability `t`, a sampled observation is digit 9;
- with probability `1 - t`, the sampled observation is one of digits 0 through 8,
  chosen uniformly.

The fine-tuning loop gradually increases `t` from 0 until it exceeds 0.5. At
each step, a small batch is sampled, the LoRA parameters are updated, and an
online diagonal Fisher estimate is refreshed for elastic weight consolidation
regularization.

## Model

The initial model is a compact multilayer perceptron for MNIST classification:

- flattened `28 x 28` input;
- small hidden layers;
- 10-way output head;
- optional LoRA adapters on all linear layers.

LoRA adapters use a low rank by default. One LoRA factor is initialized with
small random values and the other is initialized to zero, so the adapter begins
as a no-op while still permitting gradients to flow.

## Fisher Estimates

Two Fisher-related quantities are tracked for different purposes.

The EWC regularizer uses a diagonal online Fisher estimate over LoRA parameters.
This estimate is updated during fine tuning with an exponential moving average,
controlled by a configurable `rho`:

```text
F_t = (1 - rho) F_{t-1} + rho F_batch
```

Here `F_batch` is the diagonal Fisher estimate from the current fine-tuning
batch. Smaller `rho` values make the EWC weights slower and more stable; larger
values make them adapt more quickly to the current local geometry.

The final spectral analysis uses empirical Fisher matrices estimated after the
distribution shift has reached `t > 0.5`. The experiment estimates spectra for:

- the LoRA parameter subspace;
- the full model parameter space, as a control.

The LoRA spectrum is the primary object of interest. The full-model spectrum is
included to compare against the heavy skew predicted for deep network Fisher
spectra.

## Recorded Outputs

Each experiment run records time series and final spectral diagnostics,
including:

- `t` values;
- accuracy on digit 9;
- overall accuracy;
- non-9 accuracy;
- loss;
- final LoRA empirical Fisher eigenvalues;
- final full-model empirical Fisher eigenvalues;
- Fisher trace;
- Fisher condition number;
- effective rank;
- final diagonal EWC weights.

The accompanying notebook aggregates multiple runs, plots average accuracy
curves with uncertainty bands, and visualizes the average final eigenvalue
distributions for LoRA and full-model Fisher estimates.

## Data And Artifacts

MNIST data should be downloaded into `data/`. Generated datasets, model
checkpoints, cached results, and local notebook outputs are treated as runtime
artifacts rather than source files.

## Preliminary Findings

The initial experiment did not support the simple hypothesis that LoRA removes
or substantially deskews the empirical Fisher spectrum. In the observed runs,
the LoRA Fisher spectrum remained highly skewed, with a small number of dominant
eigenvalues and many near-null empirical directions.

Increasing the EWC regularization strength did not materially change this
spectral behavior. This suggests that the observed skew is not merely a failure
of weak regularization, nor simply an artifact of using the full dense parameter
space.

However, the experiment produced a more useful observation: LoRA fine tuning was
still behaviorally effective. In the initial 3-seed exploratory run, final
digit-9 accuracy reached roughly 92.5% to 95.0%, while non-9 accuracy remained
around 88.7% to 90.8%.

The Fisher spectra also suggested a much smaller active dimension than the
ambient parameter count. For the LoRA Fisher, entropy effective rank was only
about 5 to 6, while the 99% cumulative-trace dimension was about 27 to 30,
averaging 28.3 across seeds. For the full-model empirical Fisher, considering
only the nonzero sample-Gram spectrum, the 99% cumulative-trace dimension was
about 51 to 59, averaging 54.3 across seeds.

This points toward a revised hypothesis:

> Fine tuning may often occur in a low-dimensional locally identifiable
> statistical subspace, even when the ambient neural network parameter space is
> high-dimensional.

Under this interpretation, heavy Fisher skew is not necessarily a defect to be
removed. Instead, it may reveal that only a small number of local tangent
directions are statistically active for a given fine-tuning episode.

## Local Fisher Dimension

A concrete way to measure this local dimension is by the cumulative Fisher trace.
Given empirical Fisher eigenvalues sorted in descending order,

```text
lambda_1 >= lambda_2 >= ... >= lambda_r
```

define the cumulative explained Fisher variation as

```text
C(k) = (lambda_1 + ... + lambda_k) / (lambda_1 + ... + lambda_r)
```

Then define the 99% Fisher dimension as

```text
d_99 = min { k : C(k) >= 0.99 }
```

This statistic is the dimension of the smallest empirical Fisher eigenspace that
captures 99% of the observed score variance. It should be interpreted as a local,
data-distribution-dependent identifiable dimension, not as the global dimension
of the neural network or its full statistical manifold.

In the first exploratory run, this definition gave a LoRA-local Fisher dimension
near 28. That is far smaller than the LoRA parameter count, and dramatically
smaller than the full ambient parameter count. The full-model empirical Fisher
also had a small nonzero-spectral `d_99` near 54, though this estimate omits the
many exact zero eigenvalues implied by estimating the Fisher from a finite sample
in a much larger parameter space.

## Next Experiments

The next experiment should test whether this low-dimensional Fisher subspace is
not only descriptive, but operationally sufficient for fine tuning.

A proposed comparison:

| Method | Trainable directions |
| --- | --- |
| Full LoRA | All LoRA parameters |
| Top-k Fisher | Top empirical Fisher eigendirections |
| Random-k | Random subspace with the same dimension |
| Bottom-k Fisher | Low-eigenvalue Fisher directions |
| Diagonal top-k | Coordinates with largest diagonal Fisher weights |

Here `k` can be chosen using the empirical Fisher dimension, such as `d_99`.

The key behavioral tests are:

- how quickly digit-9 accuracy improves;
- how well non-9 accuracy is preserved;
- final overall accuracy;
- loss during fine tuning;
- parameter movement required to adapt;
- robustness across random seeds and sample sizes.

The strongest evidence for the revised hypothesis would be:

1. top-`d_99` Fisher subspace fine tuning performs nearly as well as full LoRA;
2. random subspaces of the same dimension perform substantially worse;
3. bottom Fisher eigenspaces perform poorly;
4. the estimated Fisher dimension remains stable across seeds and moderate
   sample-size changes.

If these hold, then the heavy Fisher skew is not merely a numerical obstacle. It
is evidence for an active local tangent subspace where information-geometric
fine-tuning methods should operate.

## References

- Ryo Karakida, Shotaro Akaho, and Shun-ichi Amari. "Universal Statistics of
  Fisher Information in Deep Neural Networks: Mean Field Approach." Proceedings
  of the Twenty-Second International Conference on Artificial Intelligence and
  Statistics, PMLR 89:1032-1041, 2019.
  https://proceedings.mlr.press/v89/karakida19a.html
