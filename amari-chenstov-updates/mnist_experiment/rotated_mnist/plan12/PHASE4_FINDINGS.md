# Plan 12 Phase 4 Findings

**Status:** Complete as of 2026-10-02T13:40:21.577266+00:00

Validated 577 of 577 frozen units; recorded compute was 8.50 hours.

## Frozen Analysis

```json
{
  "conditions": [
    {
      "chart": true,
      "current_only": false,
      "name": "gauge_no_ridge",
      "ridge_geometry": null,
      "ridge_ratio": 0.0
    },
    {
      "chart": true,
      "current_only": false,
      "name": "isotropic_ridge",
      "ridge_geometry": "isotropic",
      "ridge_ratio": 0.1
    },
    {
      "chart": true,
      "current_only": false,
      "name": "tail_ridge",
      "ridge_geometry": "tail",
      "ridge_ratio": 0.1
    },
    {
      "chart": true,
      "current_only": false,
      "name": "spectral_selector",
      "ridge_geometry": "isotropic",
      "ridge_ratio": 41.662566004060615
    }
  ],
  "paired_against_gauge_no_ridge": [
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.009339644308959281,
        "ci95_low": 0.005574717670207391,
        "mean": 0.007457180989583336,
        "positive_count": 58,
        "replicas": 64,
        "standard_deviation": 0.007683523752554878,
        "standard_error": 0.0009604404690693598
      },
      "condition": "isotropic_ridge",
      "nll_auc_gain": {
        "ci95_high": 0.11461828734706798,
        "ci95_low": 0.09377034104210456,
        "mean": 0.10419431419458627,
        "positive_count": 63,
        "replicas": 64,
        "standard_deviation": 0.04254682919380291,
        "standard_error": 0.005318353649225364
      },
      "schedule": "linear",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.019635290697759395,
        "ci95_low": -0.027458824093651666,
        "mean": -0.02354705739570553,
        "positive_count": 4,
        "replicas": 64,
        "standard_deviation": 0.015966394685494434,
        "standard_error": 0.0019957993356868042
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.007731078157902229,
        "ci95_low": 0.003904143196264388,
        "mean": 0.005817610677083309,
        "positive_count": 49,
        "replicas": 64,
        "standard_deviation": 0.0078100713502813075,
        "standard_error": 0.0009762589187851634
      },
      "condition": "tail_ridge",
      "nll_auc_gain": {
        "ci95_high": 0.10906347794631727,
        "ci95_low": 0.08818876802909423,
        "mean": 0.09862612298770575,
        "positive_count": 64,
        "replicas": 64,
        "standard_deviation": 0.042601448810659276,
        "standard_error": 0.0053251811013324095
      },
      "schedule": "linear",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.018361251374132564,
        "ci95_low": -0.025474071320117208,
        "mean": -0.021917661347124886,
        "positive_count": 4,
        "replicas": 64,
        "standard_deviation": 0.014515959073438046,
        "standard_error": 0.0018144948841797557
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.02426129421136522,
        "ci95_low": 0.01753625787196808,
        "mean": 0.02089877604166665,
        "positive_count": 63,
        "replicas": 64,
        "standard_deviation": 0.01372456395795334,
        "standard_error": 0.0017155704947441675
      },
      "condition": "spectral_selector",
      "nll_auc_gain": {
        "ci95_high": 0.20197443585017427,
        "ci95_low": 0.1711538103675745,
        "mean": 0.18656412310887438,
        "positive_count": 64,
        "replicas": 64,
        "standard_deviation": 0.06289923567877506,
        "standard_error": 0.007862404459846883
      },
      "schedule": "linear",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.10984849622696562,
        "ci95_low": -0.12420549224347406,
        "mean": -0.11702699423521984,
        "positive_count": 0,
        "replicas": 64,
        "standard_deviation": 0.029299991870425406,
        "standard_error": 0.003662498983803176
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.00914867048467468,
        "ci95_low": 0.005475392015325342,
        "mean": 0.00731203125000001,
        "positive_count": 54,
        "replicas": 64,
        "standard_deviation": 0.007496486672141505,
        "standard_error": 0.0009370608340176881
      },
      "condition": "isotropic_ridge",
      "nll_auc_gain": {
        "ci95_high": 0.12455921721248511,
        "ci95_low": 0.10042576067385792,
        "mean": 0.11249248894317151,
        "positive_count": 62,
        "replicas": 64,
        "standard_deviation": 0.049251952119647335,
        "standard_error": 0.006156494014955917
      },
      "schedule": "sigmoid",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.015070832982706133,
        "ci95_low": -0.02203370985927637,
        "mean": -0.01855227142099125,
        "positive_count": 7,
        "replicas": 64,
        "standard_deviation": 0.014209952809327013,
        "standard_error": 0.0017762441011658766
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.009804048470708829,
        "ci95_low": 0.005750235383457827,
        "mean": 0.007777141927083328,
        "positive_count": 53,
        "replicas": 64,
        "standard_deviation": 0.008273087933165307,
        "standard_error": 0.0010341359916456634
      },
      "condition": "tail_ridge",
      "nll_auc_gain": {
        "ci95_high": 0.12674946566086356,
        "ci95_low": 0.1033271475183038,
        "mean": 0.11503830658958368,
        "positive_count": 64,
        "replicas": 64,
        "standard_deviation": 0.04780064927053017,
        "standard_error": 0.005975081158816271
      },
      "schedule": "sigmoid",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.01329112967192336,
        "ci95_low": -0.02111037972313291,
        "mean": -0.017200754697528135,
        "positive_count": 11,
        "replicas": 64,
        "standard_deviation": 0.01595765316573378,
        "standard_error": 0.0019947066457167224
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.020332210312849788,
        "ci95_low": 0.013525003228816905,
        "mean": 0.016928606770833347,
        "positive_count": 57,
        "replicas": 64,
        "standard_deviation": 0.01389225935516915,
        "standard_error": 0.0017365324193961437
      },
      "condition": "spectral_selector",
      "nll_auc_gain": {
        "ci95_high": 0.20697321025128734,
        "ci95_low": 0.17138824665098806,
        "mean": 0.1891807284511377,
        "positive_count": 64,
        "replicas": 64,
        "standard_deviation": 0.07262237469448837,
        "standard_error": 0.009077796836811047
      },
      "schedule": "sigmoid",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.09512154899345268,
        "ci95_low": -0.11006577532631345,
        "mean": -0.10259366215988307,
        "positive_count": 0,
        "replicas": 64,
        "standard_deviation": 0.030498421087470962,
        "standard_error": 0.0038123026359338703
      }
    }
  ],
  "phase": "phase4",
  "pi_reference_status": "unavailable_without_a_valid_high_sample_local_reference; empirical stability and sandwich calibration only",
  "replicas": 64,
  "rows": [
    {
      "across_replica_shadow_pi_variance": 0.0006586836374252849,
      "condition": "gauge_no_ridge",
      "mean_across_replica_parameter_variance": 285.6858871573427,
      "mean_compression_relative_frobenius_error": 0.4774705344812797,
      "mean_current_accuracy_auc": 0.7496275130208333,
      "mean_current_brier_auc": 0.37236113593303677,
      "mean_current_nll_auc": 0.9329567077443244,
      "mean_final_gradient_norm": 0.8234220003062395,
      "mean_penalized_sandwich_trace": 151322.03211951596,
      "mean_resolved_displacement_squared": 0.0005853545966901427,
      "mean_total_displacement_squared": 0.17162305688264287,
      "mean_unresolved_displacement_squared": 0.17103770228595275,
      "mean_wall_seconds": 64.18266777731253,
      "mean_worst_panel_nll_auc": 1.722643345025087,
      "schedule": "linear",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.28724012546928573,
      "shadow_pi_variance": 0.007996335307185121
    },
    {
      "across_replica_shadow_pi_variance": 0.0005855618137076104,
      "condition": "isotropic_ridge",
      "mean_across_replica_parameter_variance": 256.01175578105216,
      "mean_compression_relative_frobenius_error": 0.4929877536542736,
      "mean_current_accuracy_auc": 0.7570846940104167,
      "mean_current_brier_auc": 0.3526084104198369,
      "mean_current_nll_auc": 0.8287623935497381,
      "mean_final_gradient_norm": 0.04333011503573895,
      "mean_penalized_sandwich_trace": 0.013563155468869605,
      "mean_resolved_displacement_squared": 0.0001884799780079062,
      "mean_total_displacement_squared": 0.04280716244266777,
      "mean_unresolved_displacement_squared": 0.042618682464659864,
      "mean_wall_seconds": 64.64918200037437,
      "mean_worst_panel_nll_auc": 1.5409444028445054,
      "schedule": "linear",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.26369306807358023,
      "shadow_pi_variance": 0.006845453927669051
    },
    {
      "across_replica_shadow_pi_variance": 0.0006397517649962128,
      "condition": "tail_ridge",
      "mean_across_replica_parameter_variance": 256.01492791800644,
      "mean_compression_relative_frobenius_error": 0.48928924569814986,
      "mean_current_accuracy_auc": 0.7554451236979166,
      "mean_current_brier_auc": 0.3549797907201189,
      "mean_current_nll_auc": 0.8343305847566187,
      "mean_final_gradient_norm": 0.04518924385689237,
      "mean_penalized_sandwich_trace": 0.01339328574469511,
      "mean_resolved_displacement_squared": 0.00045010559226720074,
      "mean_total_displacement_squared": 0.04355827342724664,
      "mean_unresolved_displacement_squared": 0.04310816783497944,
      "mean_wall_seconds": 66.70102677448415,
      "mean_worst_panel_nll_auc": 1.5440973116992647,
      "schedule": "linear",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.2653224641221609,
      "shadow_pi_variance": 0.007023677597866404
    },
    {
      "across_replica_shadow_pi_variance": 0.0002439837511327555,
      "condition": "spectral_selector",
      "mean_across_replica_parameter_variance": 240.9795137465663,
      "mean_compression_relative_frobenius_error": 0.4825321904884761,
      "mean_current_accuracy_auc": 0.7705262890625,
      "mean_current_brier_auc": 0.32872459330003206,
      "mean_current_nll_auc": 0.74639258463545,
      "mean_final_gradient_norm": 0.007853826216647045,
      "mean_penalized_sandwich_trace": 4.2175010638478144e-05,
      "mean_resolved_displacement_squared": 1.6449765828731405e-05,
      "mean_total_displacement_squared": 6.354118222326522e-05,
      "mean_unresolved_displacement_squared": 4.7091416394533817e-05,
      "mean_wall_seconds": 37.78054189679733,
      "mean_worst_panel_nll_auc": 1.5495686439963616,
      "schedule": "linear",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.17021313123406592,
      "shadow_pi_variance": 0.0034243344890154378
    },
    {
      "across_replica_shadow_pi_variance": 0.0007296643727055616,
      "condition": "gauge_no_ridge",
      "mean_across_replica_parameter_variance": 284.486154098745,
      "mean_compression_relative_frobenius_error": 0.4787385490176669,
      "mean_current_accuracy_auc": 0.7318640169270834,
      "mean_current_brier_auc": 0.3985785347075552,
      "mean_current_nll_auc": 1.0135995128857227,
      "mean_final_gradient_norm": 0.873241040921918,
      "mean_penalized_sandwich_trace": 143457.09560211844,
      "mean_resolved_displacement_squared": 0.0006152378359650923,
      "mean_total_displacement_squared": 0.18536141046995303,
      "mean_unresolved_displacement_squared": 0.18474617263398796,
      "mean_wall_seconds": 64.03961239303055,
      "mean_worst_panel_nll_auc": 1.7280553435011332,
      "schedule": "sigmoid",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.27567566506912494,
      "shadow_pi_variance": 0.013436204282990574
    },
    {
      "across_replica_shadow_pi_variance": 0.0006253551497390783,
      "condition": "isotropic_ridge",
      "mean_across_replica_parameter_variance": 255.33506896998767,
      "mean_compression_relative_frobenius_error": 0.5018796847209729,
      "mean_current_accuracy_auc": 0.7391760481770834,
      "mean_current_brier_auc": 0.37792049624393476,
      "mean_current_nll_auc": 0.9011070239425513,
      "mean_final_gradient_norm": 0.047545265151309954,
      "mean_penalized_sandwich_trace": 0.013322315253184933,
      "mean_resolved_displacement_squared": 0.00020018120833385653,
      "mean_total_displacement_squared": 0.04497012722331526,
      "mean_unresolved_displacement_squared": 0.0447699460149814,
      "mean_wall_seconds": 64.56740108387476,
      "mean_worst_panel_nll_auc": 1.5375935899879225,
      "schedule": "sigmoid",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.25712339364813375,
      "shadow_pi_variance": 0.012084819944224135
    },
    {
      "across_replica_shadow_pi_variance": 0.0006908798909362787,
      "condition": "tail_ridge",
      "mean_across_replica_parameter_variance": 255.36786012385147,
      "mean_compression_relative_frobenius_error": 0.4958946389899197,
      "mean_current_accuracy_auc": 0.7396411588541667,
      "mean_current_brier_auc": 0.37728702079248577,
      "mean_current_nll_auc": 0.8985612062961391,
      "mean_final_gradient_norm": 0.04712686931786721,
      "mean_penalized_sandwich_trace": 0.013029630260610934,
      "mean_resolved_displacement_squared": 0.000459246876848338,
      "mean_total_displacement_squared": 0.044925174889124224,
      "mean_unresolved_displacement_squared": 0.04446592801227589,
      "mean_wall_seconds": 66.65924381785908,
      "mean_worst_panel_nll_auc": 1.536506399925064,
      "schedule": "sigmoid",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.2584749103715968,
      "shadow_pi_variance": 0.01201026738683146
    },
    {
      "across_replica_shadow_pi_variance": 0.0003232381006733247,
      "condition": "spectral_selector",
      "mean_across_replica_parameter_variance": 240.95699374157806,
      "mean_compression_relative_frobenius_error": 0.493795307632307,
      "mean_current_accuracy_auc": 0.7487926236979167,
      "mean_current_brier_auc": 0.35694155585253956,
      "mean_current_nll_auc": 0.824418784434585,
      "mean_final_gradient_norm": 0.0083625067588135,
      "mean_penalized_sandwich_trace": 3.9074137114052024e-05,
      "mean_resolved_displacement_squared": 1.6376669750443433e-05,
      "mean_total_displacement_squared": 6.340252686159777e-05,
      "mean_unresolved_displacement_squared": 4.7025857111154333e-05,
      "mean_wall_seconds": 37.72487485610884,
      "mean_worst_panel_nll_auc": 1.5475441984427099,
      "schedule": "sigmoid",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.1730820029092419,
      "shadow_pi_variance": 0.0057051466093924535
    }
  ]
}
```

## Spectral Selector Interpretation

Fresh Phase 4 predictive gains can validate this frozen ridge value as an empirically useful strong-shrinkage condition. They do not retroactively validate the MP calibration mechanism that selected it.

Phase 4 did not include a literal no-update trajectory. Because the strong spectral condition nearly eliminates parameter displacement, its fresh gains do not yet distinguish beneficial regularized updating from preservation of the initial model.

```json
[
  {
    "schedule": "linear",
    "nll_auc_gain": {
      "ci95_high": 0.20197443585017427,
      "ci95_low": 0.1711538103675745,
      "mean": 0.18656412310887438,
      "positive_count": 64,
      "replicas": 64,
      "standard_deviation": 0.06289923567877506,
      "standard_error": 0.007862404459846883
    },
    "accuracy_auc_gain": {
      "ci95_high": 0.02426129421136522,
      "ci95_low": 0.01753625787196808,
      "mean": 0.02089877604166665,
      "positive_count": 63,
      "replicas": 64,
      "standard_deviation": 0.01372456395795334,
      "standard_error": 0.0017155704947441675
    }
  },
  {
    "schedule": "sigmoid",
    "nll_auc_gain": {
      "ci95_high": 0.20697321025128734,
      "ci95_low": 0.17138824665098806,
      "mean": 0.1891807284511377,
      "positive_count": 64,
      "replicas": 64,
      "standard_deviation": 0.07262237469448837,
      "standard_error": 0.009077796836811047
    },
    "accuracy_auc_gain": {
      "ci95_high": 0.020332210312849788,
      "ci95_low": 0.013525003228816905,
      "mean": 0.016928606770833347,
      "positive_count": 57,
      "replicas": 64,
      "standard_deviation": 0.01389225935516915,
      "standard_error": 0.0017365324193961437
    }
  }
]
```

## Warnings

- Phase 4 control limitation: the strongly regularized selector nearly stops movement, but no literal no-update trajectory was included.

