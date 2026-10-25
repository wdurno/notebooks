"""Plan 15 Phase 8: Gaussian-regression memory and resource allocation."""

PROTOCOL = "plan15-regression-v2"

SCIENTIFIC_CHANGES = [{
    "stage": "Phase 8 numerical repair",
    "change": "Trace-preserving rank-eight spectral compression",
    "reason": "Legacy Lanczos lost basis orthogonality on nearly rank-one Fisher matrices, duplicating curvature and causing an exploding archive and ridge scale.",
    "before": "Legacy Lanczos factor with clipped residual diagonal",
    "after": "Exact symmetric eigendecomposition, top-eight factor, and nonnegative residual diagonal",
    "validity": "v1 development and partial confirmation are superseded and retained. v2 uses fresh component seeds and a new scientific identity.",
    "validation": "Rank-deficient trace/diagonal conservation, repeated-update bounds, PSD and CPU/CUDA recovery are recorded in the v2 repair acceptance artifacts.",
}]
