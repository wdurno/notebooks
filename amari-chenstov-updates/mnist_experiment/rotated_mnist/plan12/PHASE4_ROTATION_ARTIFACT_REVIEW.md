# Plan 12 Phase 4 Rotation Artifact Review

## Question

Does the strong spectral condition, $\kappa/s=41.6626$, improve learning, or
does it mainly preserve the initial model while no ridge adapts to rotated
digits and forgets the normal orientation?

This review is descriptive and was motivated after inspecting the completed
Phase 4 result. It does not replace the frozen primary analysis.

## Design

The review reads all 64 paired Phase 4 replicas for each schedule. Within each
replica it compares `spectral_selector` with `gauge_no_ridge` at the same step,
using the same initialization, observations, and evaluation panel.

Positive NLL gain means lower NLL under the spectral condition. Positive
accuracy gain means higher spectral accuracy. Confidence intervals use the 64
replica-level segment means as the statistical units.

The path is $0^\circ\rightarrow30^\circ\rightarrow0^\circ\rightarrow30^\circ$.

## Rotation-Region Results

| Schedule | Region | Accuracy gain, percentage points | 95% CI | NLL gain | 95% CI |
|---|---|---:|---:|---:|---:|
| Linear | All points | 2.03 | [1.70, 2.37] | .185 | [.170, .200] |
| Linear | Near normal, $0^\circ$ to $5^\circ$ | 5.77 | [5.34, 6.21] | .225 | [.207, .243] |
| Linear | Far rotation, $25^\circ$ to $30^\circ$ | -4.74 | [-5.37, -4.10] | .106 | [.068, .144] |
| Sigmoid | All points | 1.64 | [1.30, 1.98] | .187 | [.169, .205] |
| Sigmoid | Near normal, $0^\circ$ to $5^\circ$ | 6.61 | [6.18, 7.03] | .257 | [.238, .275] |
| Sigmoid | Far rotation, $25^\circ$ to $30^\circ$ | -5.24 | [-5.90, -4.58] | .081 | [.044, .118] |

The direction-specific contrast is sharper. During the inbound leg, spectral
accuracy is ahead by 6.01 points for linear and 7.35 points for sigmoid near
normal, but behind by 2.97 and 5.26 points in the far-rotation region.

## Knot Results

| Schedule | Step | Angle | No-ridge accuracy | Spectral accuracy | Accuracy gain | NLL gain |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0 | $0^\circ$ | .8719 | .8719 | 0.00 | .000 |
| Linear | 40 | $30^\circ$ | .5940 | .5429 | -5.11 | .150 |
| Linear | 80 | $0^\circ$ | .7884 | .8677 | 7.93 | .347 |
| Linear | 120 | $30^\circ$ | .6505 | .5592 | -9.13 | -.033 |
| Sigmoid | 0 | $0^\circ$ | .8719 | .8719 | 0.00 | .000 |
| Sigmoid | 40 | $30^\circ$ | .6020 | .5442 | -5.78 | .074 |
| Sigmoid | 80 | $0^\circ$ | .7863 | .8676 | 8.12 | .348 |
| Sigmoid | 120 | $30^\circ$ | .6568 | .5600 | -9.67 | -.104 |

At the return to $0^\circ$, the spectral condition has lost only about .4
accuracy points from initialization, while no ridge has lost about 8.4 to 8.6
points. At the final $30^\circ$ state, no ridge is ahead by 9.1 to 9.7 points.

## Is There Any Learning?

At matched angles on the second outbound pass, no-ridge accuracy improves over
the first pass by 3.76 points for linear and 4.01 points for sigmoid. The
spectral condition improves by only 1.31 and 1.22 points. NLL improves on the
second pass under both conditions, but the improvement is about three to four
times larger without ridge.

The spectral condition therefore is not exactly static, but it learns much
less. This agrees with its total squared displacement being only .034% to
.037% of no ridge.

## Interpretation

The accuracy evidence strongly supports the preservation hypothesis. The
large aggregate accuracy gain mostly rewards resistance to forgetting near the
original orientation, while the condition adapts poorly at the largest
rotation.

The NLL result is more nuanced. Spectral NLL remains better over most points,
including the pooled far-rotation region, even when its accuracy is worse.
This is compatible with fewer correct classifications but less severe
overconfidence. By the final $30^\circ$ point, spectral NLL is also worse for
sigmoid and is directionally worse for linear.

Because Phase 4 omitted a literal no-update trajectory, these artifacts cannot
separate beneficial tiny-neighborhood learning from simple preservation of the
initial estimator. A paired no-update control is the clean next test.

## Terminology

- **Gauge:** adding the same feature-weight vector and bias to every class
  logit leaves softmax probabilities unchanged. These 25 nonidentifiable
  directions are removed by the gauge-fixed 487-coordinate chart.
- **Scale $s$:** $s=\operatorname{tr}(\widehat F)/487$, the mean Fisher
  eigenvalue in the gauge-fixed chart. The ratio $\kappa/s$ is dimensionless.
- **PIT:** the empirical CDF of resampled pseudo-tail maxima evaluated at a
  held-out observed maximum. Uniform PIT values would have mean $.5$ under an
  ideal independent calibration experiment.
- **Top-eight trace fraction:** the sum of the largest eight nonnegative Fisher
  eigenvalues divided by the total nonnegative Fisher trace.
- **Fixed-anchor batch variance:** variation across independent four-example
  optimization branches while holding an anchor state fixed.
- **Isotropic ridge:** $\widehat F+\kappa I$.
- **Tail ridge:** $\widehat F+\kappa(I-U_8U_8^T)$.
- **Spectral selector:** isotropic ridge using the frozen median of eight
  checkpoint-specific empirical $.99$ tail-maximum ratios.
- **Tail-spectrum edge diagnostic:** resampled and deformed-MP maxima after
  projecting scores off the leading eight Fisher directions.
- **Pooled recommendation variance:** variance after pooling all shadow
  $\widehat\pi^\star$ recommendations over times and replicas; it is not a
  pure within-trajectory temporal variance.
