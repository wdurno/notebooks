# Rotated-MNIST agent notes

This directory is the detachable implementation boundary for
[../plan5.md](../plan5.md), its oracle-calibration follow-up
[../plan6.md](../plan6.md), and the artifact-only anchor audit in
[../plan7.md](../plan7.md). Follow the repository and MNIST experiment
instructions, with these additional constraints:

- Rotation angle is the only changing environmental coordinate. Do not reuse
  `p` or reinterpret any historical mixture field.
- The initializer contains all ten upright MNIST classes.
- Apply deterministic bilinear rotation to float tensors before model input.
  The canonical pipeline currently has no additional normalization.
- Keep all Plan 5 schemas, runners, transforms, schedules, and loaders in this
  package unless a check-in explicitly approves a shared helper.
- Existing runners must never import this package.
- Generated artifacts belong only under
  `cache/mnist_experiment/rotated_mnist/`.
- Notebooks load completed artifacts only. They never train, download, resume,
  or repair runs.

The accepted principal schedule is
`0 -> 15 -> 30 -> 0 -> 15 -> 30` degrees. Its return leg is both a reversal
and a two-times speed challenge under the 20-transitions-per-arrow contract.
Use matched first and second ascents for the clean revisitation comparison.
