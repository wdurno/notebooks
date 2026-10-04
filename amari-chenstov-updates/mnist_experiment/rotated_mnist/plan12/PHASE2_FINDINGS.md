# Plan 12 Phase 2 Findings

**Status:** Complete as of 2026-10-02T13:40:21.577266+00:00

Validated 705 of 705 frozen units; recorded compute was 10.92 hours.

## Frozen Analysis

```json
{
  "conditions": [
    {
      "chart": true,
      "current_only": true,
      "name": "current_only",
      "ridge_geometry": null,
      "ridge_ratio": 0.0
    },
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
      "chart": false,
      "current_only": false,
      "name": "legacy_raw",
      "ridge_geometry": null,
      "ridge_ratio": 0.0
    }
  ],
  "paired_against_gauge_no_ridge": [
    {
      "accuracy_auc_gain": {
        "ci95_high": -0.2067809793795886,
        "ci95_low": -0.2305640596829114,
        "mean": -0.21867251953125,
        "positive_count": 0,
        "replicas": 64,
        "standard_deviation": 0.04853689857820989,
        "standard_error": 0.0060671123222762365
      },
      "condition": "current_only",
      "nll_auc_gain": {
        "ci95_high": -6.908793201426842,
        "ci95_low": -9.042412975794242,
        "mean": -7.9756030886105425,
        "positive_count": 0,
        "replicas": 64,
        "standard_deviation": 4.354326070137555,
        "standard_error": 0.5442907587671943
      },
      "schedule": "linear",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.22678473479665467,
        "ci95_low": -0.24030134814898146,
        "mean": -0.23354304147281807,
        "positive_count": 0,
        "replicas": 64,
        "standard_deviation": 0.027584925208830206,
        "standard_error": 0.003448115651103776
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.008186009868668367,
        "ci95_low": 0.004569367735498309,
        "mean": 0.006377688802083338,
        "positive_count": 53,
        "replicas": 64,
        "standard_deviation": 0.007380902312591957,
        "standard_error": 0.0009226127890739946
      },
      "condition": "isotropic_ridge",
      "nll_auc_gain": {
        "ci95_high": 0.10778675434924946,
        "ci95_low": 0.08891017298905747,
        "mean": 0.09834846366915347,
        "positive_count": 64,
        "replicas": 64,
        "standard_deviation": 0.038523635428963254,
        "standard_error": 0.004815454428620407
      },
      "schedule": "linear",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.016143944448805292,
        "ci95_low": -0.024124242007052564,
        "mean": -0.020134093227928928,
        "positive_count": 4,
        "replicas": 64,
        "standard_deviation": 0.016286321547443406,
        "standard_error": 0.0020357901934304258
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.00832183281806663,
        "ci95_low": 0.004836773952766705,
        "mean": 0.006579303385416667,
        "positive_count": 54,
        "replicas": 64,
        "standard_deviation": 0.007112365031224337,
        "standard_error": 0.0008890456289030421
      },
      "condition": "tail_ridge",
      "nll_auc_gain": {
        "ci95_high": 0.1103490736046766,
        "ci95_low": 0.091208278120473,
        "mean": 0.1007786758625748,
        "positive_count": 64,
        "replicas": 64,
        "standard_deviation": 0.039062847926946136,
        "standard_error": 0.004882855990868267
      },
      "schedule": "linear",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.017008527581678562,
        "ci95_low": -0.025847094968978357,
        "mean": -0.02142781127532846,
        "positive_count": 10,
        "replicas": 64,
        "standard_deviation": 0.018037892627142442,
        "standard_error": 0.0022547365783928052
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.0007345077130092071,
        "ci95_low": -0.0019953150046758622,
        "mean": -0.0006304036458333277,
        "positive_count": 31,
        "replicas": 64,
        "standard_deviation": 0.0055710667707858565,
        "standard_error": 0.0006963833463482321
      },
      "condition": "legacy_raw",
      "nll_auc_gain": {
        "ci95_high": 0.000917995032972312,
        "ci95_low": -0.017235534583931574,
        "mean": -0.00815876977547963,
        "positive_count": 27,
        "replicas": 64,
        "standard_deviation": 0.03704801962633446,
        "standard_error": 0.004631002453291807
      },
      "schedule": "linear",
      "shadow_pi_mean_shift": {
        "ci95_high": 0.008196168055294885,
        "ci95_low": 0.0010271752859642032,
        "mean": 0.004611671670629544,
        "positive_count": 37,
        "replicas": 64,
        "standard_deviation": 0.014630597488429964,
        "standard_error": 0.0018288246860537454
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": -0.2065075462189843,
        "ci95_low": -0.2513276360726823,
        "mean": -0.22891759114583332,
        "positive_count": 0,
        "replicas": 64,
        "standard_deviation": 0.09146957112999603,
        "standard_error": 0.011433696391249503
      },
      "condition": "current_only",
      "nll_auc_gain": {
        "ci95_high": -7.459934766682815,
        "ci95_low": -8.984987559256162,
        "mean": -8.222461162969488,
        "positive_count": 0,
        "replicas": 64,
        "standard_deviation": 3.112352637904788,
        "standard_error": 0.3890440797380985
      },
      "schedule": "sigmoid",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.21472340463510126,
        "ci95_low": -0.22846932163481048,
        "mean": -0.22159636313495587,
        "positive_count": 0,
        "replicas": 64,
        "standard_deviation": 0.028052891836141302,
        "standard_error": 0.0035066114795176627
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.010146819717080398,
        "ci95_low": 0.005972125595419592,
        "mean": 0.008059472656249995,
        "positive_count": 53,
        "replicas": 64,
        "standard_deviation": 0.008519783921756749,
        "standard_error": 0.0010649729902195936
      },
      "condition": "isotropic_ridge",
      "nll_auc_gain": {
        "ci95_high": 0.1259603067334451,
        "ci95_low": 0.10529850752729136,
        "mean": 0.11562940713036823,
        "positive_count": 64,
        "replicas": 64,
        "standard_deviation": 0.04216693715541577,
        "standard_error": 0.0052708671444269715
      },
      "schedule": "sigmoid",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.021209805314933203,
        "ci95_low": -0.02955121449602053,
        "mean": -0.025380509905476866,
        "positive_count": 3,
        "replicas": 64,
        "standard_deviation": 0.017023284043035367,
        "standard_error": 0.002127910505379421
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.009356688037882867,
        "ci95_low": 0.005672569774617106,
        "mean": 0.007514628906249987,
        "positive_count": 55,
        "replicas": 64,
        "standard_deviation": 0.00751860870054237,
        "standard_error": 0.0009398260875677963
      },
      "condition": "tail_ridge",
      "nll_auc_gain": {
        "ci95_high": 0.12383134703173679,
        "ci95_low": 0.10499462754055522,
        "mean": 0.114412987286146,
        "positive_count": 64,
        "replicas": 64,
        "standard_deviation": 0.03844228467588078,
        "standard_error": 0.004805285584485098
      },
      "schedule": "sigmoid",
      "shadow_pi_mean_shift": {
        "ci95_high": -0.017145557014557743,
        "ci95_low": -0.02523888812421851,
        "mean": -0.021192222569388127,
        "positive_count": 7,
        "replicas": 64,
        "standard_deviation": 0.016517002264613806,
        "standard_error": 0.0020646252830767257
      }
    },
    {
      "accuracy_auc_gain": {
        "ci95_high": 0.0013266273277365882,
        "ci95_low": -0.0013628382652365913,
        "mean": -1.810546875000152e-05,
        "positive_count": 28,
        "replicas": 64,
        "standard_deviation": 0.005488705291781999,
        "standard_error": 0.0006860881614727498
      },
      "condition": "legacy_raw",
      "nll_auc_gain": {
        "ci95_high": 0.00941100917559896,
        "ci95_low": -0.008622065618889195,
        "mean": 0.0003944717783548827,
        "positive_count": 32,
        "replicas": 64,
        "standard_deviation": 0.03680219345813909,
        "standard_error": 0.004600274182267387
      },
      "schedule": "sigmoid",
      "shadow_pi_mean_shift": {
        "ci95_high": 0.0058652530107511855,
        "ci95_low": -0.0015757426084942704,
        "mean": 0.0021447552011284576,
        "positive_count": 37,
        "replicas": 64,
        "standard_deviation": 0.015185705345398889,
        "standard_error": 0.0018982131681748611
      }
    }
  ],
  "phase": "phase2",
  "pi_reference_status": "unavailable_without_a_valid_high_sample_local_reference; empirical stability and sandwich calibration only",
  "replicas": 64,
  "rows": [
    {
      "across_replica_shadow_pi_variance": 1.6809880052490447e-05,
      "condition": "current_only",
      "mean_across_replica_parameter_variance": 2149.985764067397,
      "mean_compression_relative_frobenius_error": 0.4841308441472092,
      "mean_current_accuracy_auc": 0.5291022916666667,
      "mean_current_brier_auc": 0.8667043343221085,
      "mean_current_nll_auc": 8.915414017235578,
      "mean_final_gradient_norm": 4.368251681399542e-05,
      "mean_penalized_sandwich_trace": 11566209948.115776,
      "mean_resolved_displacement_squared": 2.511809180504385,
      "mean_total_displacement_squared": 506.56979669321277,
      "mean_unresolved_displacement_squared": 504.05798751270845,
      "mean_wall_seconds": 42.04742841525017,
      "mean_worst_panel_nll_auc": 11.182542855176484,
      "schedule": "linear",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.05420734390770666,
      "shadow_pi_variance": 0.00020527823545263027
    },
    {
      "across_replica_shadow_pi_variance": 0.000705423110490726,
      "condition": "gauge_no_ridge",
      "mean_across_replica_parameter_variance": 297.69006102445877,
      "mean_compression_relative_frobenius_error": 0.46371263986358247,
      "mean_current_accuracy_auc": 0.7477748111979167,
      "mean_current_brier_auc": 0.37560557480348356,
      "mean_current_nll_auc": 0.9398109286250352,
      "mean_final_gradient_norm": 0.8233282729497357,
      "mean_penalized_sandwich_trace": 162679.58387079826,
      "mean_resolved_displacement_squared": 0.0006060004712181324,
      "mean_total_displacement_squared": 0.17250439294803918,
      "mean_unresolved_displacement_squared": 0.17189839247682107,
      "mean_wall_seconds": 64.78578328237535,
      "mean_worst_panel_nll_auc": 1.7421008629523995,
      "schedule": "linear",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.2877503853805247,
      "shadow_pi_variance": 0.007732195937640611
    },
    {
      "across_replica_shadow_pi_variance": 0.0004995994023688331,
      "condition": "isotropic_ridge",
      "mean_across_replica_parameter_variance": 267.2831371636143,
      "mean_compression_relative_frobenius_error": 0.48090437364223704,
      "mean_current_accuracy_auc": 0.7541525,
      "mean_current_brier_auc": 0.3574684075991457,
      "mean_current_nll_auc": 0.8414624649558818,
      "mean_final_gradient_norm": 0.04312032188083777,
      "mean_penalized_sandwich_trace": 0.013817679807814868,
      "mean_resolved_displacement_squared": 0.00020470564756646252,
      "mean_total_displacement_squared": 0.041919988648568184,
      "mean_unresolved_displacement_squared": 0.041715283001001725,
      "mean_wall_seconds": 65.4920344636721,
      "mean_worst_panel_nll_auc": 1.5685015676242144,
      "schedule": "linear",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.2676162921525958,
      "shadow_pi_variance": 0.00679815079471824
    },
    {
      "across_replica_shadow_pi_variance": 0.000560572267406145,
      "condition": "tail_ridge",
      "mean_across_replica_parameter_variance": 267.2411274977531,
      "mean_compression_relative_frobenius_error": 0.4810292972594628,
      "mean_current_accuracy_auc": 0.7543541145833333,
      "mean_current_brier_auc": 0.3569791822849126,
      "mean_current_nll_auc": 0.8390322527624604,
      "mean_final_gradient_norm": 0.044107202621338125,
      "mean_penalized_sandwich_trace": 0.014183375102672703,
      "mean_resolved_displacement_squared": 0.0004562436789033287,
      "mean_total_displacement_squared": 0.041947936539955204,
      "mean_unresolved_displacement_squared": 0.04149169286105188,
      "mean_wall_seconds": 67.4869967516097,
      "mean_worst_panel_nll_auc": 1.570497368913383,
      "schedule": "linear",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.2663225741051963,
      "shadow_pi_variance": 0.006720796397302646
    },
    {
      "across_replica_shadow_pi_variance": 0.0007867801124694146,
      "condition": "legacy_raw",
      "mean_across_replica_parameter_variance": null,
      "mean_compression_relative_frobenius_error": 0.45939784593944566,
      "mean_current_accuracy_auc": 0.7471444075520833,
      "mean_current_brier_auc": 0.37700662090096554,
      "mean_current_nll_auc": 0.9479696984005149,
      "mean_final_gradient_norm": 0.7932001101099407,
      "mean_penalized_sandwich_trace": 153572.90346038408,
      "mean_resolved_displacement_squared": 0.0005157093243808707,
      "mean_total_displacement_squared": 0.16955868624905118,
      "mean_unresolved_displacement_squared": 0.16904297692467032,
      "mean_wall_seconds": 61.34470684251541,
      "mean_worst_panel_nll_auc": 1.752421899330422,
      "schedule": "linear",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.2923620570511543,
      "shadow_pi_variance": 0.008195327196130042
    },
    {
      "across_replica_shadow_pi_variance": 1.341165442064444e-05,
      "condition": "current_only",
      "mean_across_replica_parameter_variance": 4142.458566522557,
      "mean_compression_relative_frobenius_error": 0.4856022994497409,
      "mean_current_accuracy_auc": 0.5012449348958333,
      "mean_current_brier_auc": 0.8971525235797367,
      "mean_current_nll_auc": 9.243233845228684,
      "mean_final_gradient_norm": 0.0005638614103993266,
      "mean_penalized_sandwich_trace": 6178564812.098927,
      "mean_resolved_displacement_squared": 6.747854608929292,
      "mean_total_displacement_squared": 404.81434638090883,
      "mean_unresolved_displacement_squared": 398.0664917719795,
      "mean_wall_seconds": 42.484882793843866,
      "mean_worst_panel_nll_auc": 11.202022505851867,
      "schedule": "sigmoid",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.057789176327257136,
      "shadow_pi_variance": 0.0007034808341124369
    },
    {
      "across_replica_shadow_pi_variance": 0.0007756699739799736,
      "condition": "gauge_no_ridge",
      "mean_across_replica_parameter_variance": 295.5512897978329,
      "mean_compression_relative_frobenius_error": 0.46822978606088717,
      "mean_current_accuracy_auc": 0.7301625260416666,
      "mean_current_brier_auc": 0.40161931735519824,
      "mean_current_nll_auc": 1.0207726822591947,
      "mean_final_gradient_norm": 0.8758132783209173,
      "mean_penalized_sandwich_trace": 156354.45436745262,
      "mean_resolved_displacement_squared": 0.0006321812947368177,
      "mean_total_displacement_squared": 0.18173132796327812,
      "mean_unresolved_displacement_squared": 0.18109914666854132,
      "mean_wall_seconds": 64.86946892168766,
      "mean_worst_panel_nll_auc": 1.7378253908578587,
      "schedule": "sigmoid",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.279385539462213,
      "shadow_pi_variance": 0.014569205471557184
    },
    {
      "across_replica_shadow_pi_variance": 0.000583735174905265,
      "condition": "isotropic_ridge",
      "mean_across_replica_parameter_variance": 266.63983088105266,
      "mean_compression_relative_frobenius_error": 0.4875151195731206,
      "mean_current_accuracy_auc": 0.7382219986979166,
      "mean_current_brier_auc": 0.37963702586650877,
      "mean_current_nll_auc": 0.9051432751288265,
      "mean_final_gradient_norm": 0.044310415983090934,
      "mean_penalized_sandwich_trace": 0.013775850405995384,
      "mean_resolved_displacement_squared": 0.00019711314260152898,
      "mean_total_displacement_squared": 0.04380295302587806,
      "mean_unresolved_displacement_squared": 0.04360583988327653,
      "mean_wall_seconds": 65.43290784106271,
      "mean_worst_panel_nll_auc": 1.5513371197379768,
      "schedule": "sigmoid",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.25400502955673615,
      "shadow_pi_variance": 0.01201299462688014
    },
    {
      "across_replica_shadow_pi_variance": 0.0005908504483363847,
      "condition": "tail_ridge",
      "mean_across_replica_parameter_variance": 266.3773718823418,
      "mean_compression_relative_frobenius_error": 0.4904402119337789,
      "mean_current_accuracy_auc": 0.7376771549479166,
      "mean_current_brier_auc": 0.38056602219660535,
      "mean_current_nll_auc": 0.9063596949730488,
      "mean_final_gradient_norm": 0.04601856997667255,
      "mean_penalized_sandwich_trace": 0.013582445900454683,
      "mean_resolved_displacement_squared": 0.00046354571710161165,
      "mean_total_displacement_squared": 0.04372552352861704,
      "mean_unresolved_displacement_squared": 0.04326197781151543,
      "mean_wall_seconds": 67.43852101229665,
      "mean_worst_panel_nll_auc": 1.5564811566594492,
      "schedule": "sigmoid",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.2581933168928249,
      "shadow_pi_variance": 0.012923511931105093
    },
    {
      "across_replica_shadow_pi_variance": 0.0007255476834066083,
      "condition": "legacy_raw",
      "mean_across_replica_parameter_variance": null,
      "mean_compression_relative_frobenius_error": 0.4714926061503295,
      "mean_current_accuracy_auc": 0.7301444205729166,
      "mean_current_brier_auc": 0.401453021484558,
      "mean_current_nll_auc": 1.0203782104808399,
      "mean_final_gradient_norm": 0.868683368121189,
      "mean_penalized_sandwich_trace": 152361.7606396034,
      "mean_resolved_displacement_squared": 0.0006153017819334351,
      "mean_total_displacement_squared": 0.17954641520033612,
      "mean_unresolved_displacement_squared": 0.17893111341840268,
      "mean_wall_seconds": 61.33278995968746,
      "mean_worst_panel_nll_auc": 1.7335372067943542,
      "schedule": "sigmoid",
      "shadow_pi_boundary_fraction": 0.008333333333333333,
      "shadow_pi_mean": 0.28153029466334145,
      "shadow_pi_variance": 0.014390584382092678
    }
  ]
}
```

