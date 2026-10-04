# Plan 12 Phase 3 Findings

**Status:** Complete as of 2026-10-02T13:40:21.577266+00:00

Validated 9 of 9 frozen units; recorded compute was 0.00 hours.

## Frozen Analysis

```json
{
  "classification": "empirical_selector",
  "failed_checkpoint_count": 0,
  "fallback": false,
  "selected_scale_ratio": 41.662566004060615,
  "selection_source": "empirical_99_percent_tail_maximum",
  "valid_checkpoint_count": 8
}
```

## Scientific Contract Audit

**Selector status:** `failed_not_calibrated_or_deployable`

The frozen ratio 41.6626 lies outside the Phase 1 useful isotropic interval [0.1, 1.0]; this violates the preregistered selector contract.

The values are a descriptive dependence-sensitive diagnostic, not eight independent observations for a nominal coverage test. Their concentration near zero is nevertheless inconsistent with a reassuring calibration story.

Fresh Phase 4 predictive gains can validate this frozen ridge value as an empirically useful strong-shrinkage condition. They do not retroactively validate the MP calibration mechanism that selected it.

## Warnings

- Phase 3 scientific contract: the spectral selector is not calibrated or deployable because its frozen ratio lies outside the Phase 1 useful region.

