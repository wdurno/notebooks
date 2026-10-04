# Plan 12 Phase 1 Findings

**Status:** Complete as of 2026-10-02T13:40:21.577266+00:00

Validated 4236 of 4236 frozen units; recorded compute was 0.41 hours.

## Frozen Analysis

```json
{
  "baseline": {
    "condition": "gauge_no_ridge",
    "current_accuracy": 0.7704391479492188,
    "current_nll": 0.8539036805887008,
    "current_reference_valid_fraction": 0.0,
    "fisher_total_bias_squared": null,
    "fisher_total_mse": null,
    "fisher_variance": 0.5745606852825407,
    "functional_total_mse": null,
    "heldout_current_plus_retention_nll": 2.7670499855448725,
    "parameter_variance": 0.14261300236621396,
    "probability_total_mse": null,
    "ridge_geometry": null,
    "ridge_ratio": 0.0,
    "target_valid_fraction": 0.0,
    "worst_retention_nll": 1.9131463049561717
  },
  "isotropic": {
    "classification": "promising_contiguous_region",
    "eligible_improving_conditions": [
      "isotropic_1em01",
      "isotropic_1ep00"
    ],
    "minimum_relative_improvement": 0.005,
    "practical_tie_relative_tolerance": 0.005,
    "selected_condition": "isotropic_1em01",
    "selected_ratio": 0.1,
    "selection_metric": "mean_heldout_current_plus_worst_retention_nll"
  },
  "tail": {
    "classification": "promising_contiguous_region",
    "eligible_improving_conditions": [
      "tail_1em01",
      "tail_1ep00"
    ],
    "minimum_relative_improvement": 0.005,
    "practical_tie_relative_tolerance": 0.005,
    "selected_condition": "tail_1em01",
    "selected_ratio": 0.1,
    "selection_metric": "mean_heldout_current_plus_worst_retention_nll"
  }
}
```

## Warnings

- Phase 3 scientific contract: the spectral selector is not calibrated or deployable because its frozen ratio lies outside the Phase 1 useful region.

