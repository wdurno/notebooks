# Plan 12 Phase 5 Findings

**Status:** Complete exploratory probe

The immutable sample contains 16 paired replicas per schedule at $\kappa_t/s_t=11.818529$.

Post hoc descriptive probe. Confidence intervals describe paired uncertainty at the frozen 16-replica size and are not selector-calibration or confirmatory claims.

## Paired Results

### Linear

Against `gauge_no_ridge`:

- NLL-AUC gain: 0.186234 (95% CI [0.152037, 0.220432]; 16/16 favorable).
- Accuracy-AUC gain: 0.0188158 (95% CI [0.0134326, 0.0241991]; 16/16 favorable).
- Cumulative squared displacement ratio: 0.00310023 (95% CI [0.00254778, 0.00365269]).

Against `spectral_selector`:

- NLL-AUC gain: 0.0126172 (95% CI [0.00690237, 0.0183321]; 13/16 favorable).
- Accuracy-AUC gain: 0.00273951 (95% CI [0.000731654, 0.00474736]; 11/16 favorable).
- Cumulative squared displacement ratio: 7.74053 (95% CI [7.56758, 7.91348]).

### Sigmoid

Against `gauge_no_ridge`:

- NLL-AUC gain: 0.199649 (95% CI [0.157421, 0.241877]; 16/16 favorable).
- Accuracy-AUC gain: 0.0169292 (95% CI [0.0108629, 0.0229956]; 15/16 favorable).
- Cumulative squared displacement ratio: 0.00294121 (95% CI [0.00239414, 0.00348828]).

Against `spectral_selector`:

- NLL-AUC gain: 0.0182355 (95% CI [0.0130124, 0.0234585]; 16/16 favorable).
- Accuracy-AUC gain: 0.00384401 (95% CI [0.0025542, 0.00513382]; 15/16 favorable).
- Cumulative squared displacement ratio: 7.62956 (95% CI [7.43196, 7.82716]).

## Artifact Review

The probe found a reproducible middle ground relative to the two tested
controls. Compared with the $q_{.99}$ condition, $q_{.01}$ improved mean
NLL-AUC and accuracy-AUC under both schedules. Its cumulative squared movement
was about 7.7 times larger, so the predictive change is accompanied by more
learner adaptation rather than being an evaluation-only fluctuation.

The angle-stratified result clarifies the tradeoff. Relative to $q_{.99}$,
$q_{.01}$ was slightly worse near $0$--$5^\circ$: mean NLL gain was $-.00645$
for linear and $-.00632$ for sigmoid, while mean accuracy gain was about
$-.00275$ under both schedules. At $25$--$30^\circ$, it was better: mean NLL
gain was $.0550$ for linear and $.0587$ for sigmoid, and mean accuracy gain
was $.0150$ and $.0147$, respectively. The lower quantile therefore exchanges
a small amount of upright-state protection for measurably better rotated-state
adaptation.

This is not unrestricted learning. The $q_{.01}$ condition retained only
$.310\%$ of no-ridge cumulative squared movement under linear scheduling and
$.294\%$ under sigmoid scheduling. It also remained less accurate than no
added ridge at $25$--$30^\circ$ by $.0344$ and $.0417$, respectively, despite
lower NLL there. The result supports weakening $q_{.99}$, but does not show
that $q_{.01}$ reaches the best bias-adaptation balance.

Optimization was healthy under $q_{.01}$: both schedules averaged about 11.1
LBFGS iterations, no update reached the 50-iteration cap, and mean relative
final-gradient norm was about $.0016$. By contrast, the matched no-ridge
control reached the cap on more than $99.5\%$ of updates. Fisher compression
error remained substantial at about $.51$--$.52$, but was similar in scale to
both controls rather than uniquely induced by $q_{.01}$.

These are post hoc descriptive findings. They do not validate the MP selector,
establish nominal frequentist coverage, or remove the Phase 4 limitation that
no literal no-update control was included. They do provide evidence that the
bootstrap edge contains useful regularization-scale information below its
$q_{.99}$ extreme.

## Full Artifact Summary

```json
{
  "condition": {
    "chart": true,
    "current_only": false,
    "name": "mp_q01_isotropic",
    "ridge_geometry": "isotropic",
    "ridge_ratio": 11.818529434984821
  },
  "controls": [
    "gauge_no_ridge",
    "spectral_selector"
  ],
  "interpretation_contract": "Post hoc descriptive probe. Confidence intervals describe paired uncertainty at the frozen 16-replica size and are not selector-calibration or confirmatory claims.",
  "mp_quantile": 0.01,
  "phase": "phase5",
  "quantile_method": "higher",
  "replicas_per_schedule": 16,
  "ridge_ratio": 11.818529434984821,
  "schedule_results": [
    {
      "comparisons": [
        {
          "accuracy_auc_gain": {
            "ci95_high": 0.024199065161140676,
            "ci95_low": 0.013432601505526012,
            "favorable_count": 16,
            "mean": 0.018815833333333344,
            "median": 0.017878333333333274,
            "replicas": 16,
            "standard_deviation": 0.01010248042516004,
            "standard_error": 0.00252562010629001
          },
          "brier_auc_gain": {
            "ci95_high": 0.04983738069493421,
            "ci95_low": 0.03201191091700138,
            "favorable_count": 16,
            "mean": 0.040924645805967795,
            "median": 0.03825246107751648,
            "replicas": 16,
            "standard_deviation": 0.0167261475319184,
            "standard_error": 0.0041815368829796
          },
          "control": "gauge_no_ridge",
          "cumulative_squared_displacement_ratio": {
            "ci95_high": 0.0036526883115137274,
            "ci95_low": 0.0025477783653875593,
            "favorable_count": 16,
            "mean": 0.0031002333384506434,
            "median": 0.002941887357660652,
            "replicas": 16,
            "standard_deviation": 0.0010367685675958364,
            "standard_error": 0.0002591921418989591
          },
          "nll_auc_gain": {
            "ci95_high": 0.2204316879711852,
            "ci95_low": 0.1520367465940621,
            "favorable_count": 16,
            "mean": 0.18623421728262365,
            "median": 0.17958682898700234,
            "replicas": 16,
            "standard_deviation": 0.06417692740568746,
            "standard_error": 0.016044231851421866
          },
          "segments": {
            "far_rotation_25_to_30_degrees": {
              "accuracy_gain": {
                "ci95_high": -0.02369232759110458,
                "ci95_low": -0.045085797408895456,
                "favorable_count": 1,
                "mean": -0.03438906250000002,
                "median": -0.033279999999999976,
                "replicas": 16,
                "standard_deviation": 0.020074103900195227,
                "standard_error": 0.005018525975048807
              },
              "nll_gain": {
                "ci95_high": 0.23329394341937698,
                "ci95_low": 0.07550357508309685,
                "favorable_count": 13,
                "mean": 0.15439875925123692,
                "median": 0.17911387132763856,
                "replicas": 16,
                "standard_deviation": 0.1480592103763579,
                "standard_error": 0.03701480259408948
              }
            },
            "first_ascent": {
              "accuracy_gain": {
                "ci95_high": 0.04756724956901459,
                "ci95_low": 0.027496812930985368,
                "favorable_count": 16,
                "mean": 0.03753203124999998,
                "median": 0.03515124999999997,
                "replicas": 16,
                "standard_deviation": 0.018832664071119214,
                "standard_error": 0.004708166017779803
              },
              "nll_gain": {
                "ci95_high": 0.3141163086046667,
                "ci95_low": 0.19330418672289534,
                "favorable_count": 16,
                "mean": 0.25371024766378103,
                "median": 0.23532492877244943,
                "replicas": 16,
                "standard_deviation": 0.11336146533092677,
                "standard_error": 0.02834036633273169
              }
            },
            "near_upright_0_to_5_degrees": {
              "accuracy_gain": {
                "ci95_high": 0.0546260037836043,
                "ci95_low": 0.041434621216395684,
                "favorable_count": 16,
                "mean": 0.04803031249999999,
                "median": 0.04674000000000006,
                "replicas": 16,
                "standard_deviation": 0.012377851115163981,
                "standard_error": 0.0030944627787909953
              },
              "nll_gain": {
                "ci95_high": 0.2270709928826305,
                "ci95_low": 0.16800703677355622,
                "favorable_count": 16,
                "mean": 0.19753901482809336,
                "median": 0.19731739052757621,
                "replicas": 16,
                "standard_deviation": 0.055421397360428795,
                "standard_error": 0.013855349340107199
              }
            },
            "return_leg": {
              "accuracy_gain": {
                "ci95_high": 0.016333069689243626,
                "ci95_low": 0.006411930310756376,
                "favorable_count": 14,
                "mean": 0.0113725,
                "median": 0.010881249999999953,
                "replicas": 16,
                "standard_deviation": 0.009309288506647527,
                "standard_error": 0.0023273221266618817
              },
              "nll_gain": {
                "ci95_high": 0.18529889958896278,
                "ci95_low": 0.11340680323602097,
                "favorable_count": 16,
                "mean": 0.14935285141249188,
                "median": 0.1327265491026639,
                "replicas": 16,
                "standard_deviation": 0.06745840782646939,
                "standard_error": 0.016864601956617348
              }
            },
            "second_ascent": {
              "accuracy_gain": {
                "ci95_high": 0.014164627013490295,
                "ci95_low": -0.0012374928671487935,
                "favorable_count": 13,
                "mean": 0.006463567073170751,
                "median": 0.009279268292682885,
                "replicas": 16,
                "standard_deviation": 0.014452249092856055,
                "standard_error": 0.0036130622732140137
              },
              "nll_gain": {
                "ci95_high": 0.19489323148236967,
                "ci95_low": 0.1092439679872316,
                "favorable_count": 15,
                "mean": 0.15206859973480064,
                "median": 0.15422690594370775,
                "replicas": 16,
                "standard_deviation": 0.08036715077171812,
                "standard_error": 0.02009178769292953
              }
            }
          },
          "worst_panel_nll_auc_gain": {
            "ci95_high": 0.2912636928317048,
            "ci95_low": 0.14400395398109195,
            "favorable_count": 15,
            "mean": 0.21763382340639836,
            "median": 0.21062652764320378,
            "replicas": 16,
            "standard_deviation": 0.13817801989018677,
            "standard_error": 0.03454450497254669
          }
        },
        {
          "accuracy_auc_gain": {
            "ci95_high": 0.004747356520622285,
            "ci95_low": 0.0007316538960443946,
            "favorable_count": 11,
            "mean": 0.00273950520833334,
            "median": 0.0033214583333333048,
            "replicas": 16,
            "standard_deviation": 0.003768048493518633,
            "standard_error": 0.0009420121233796583
          },
          "brier_auc_gain": {
            "ci95_high": 0.00623032729209229,
            "ci95_low": 0.0011725321988485578,
            "favorable_count": 13,
            "mean": 0.003701429745470424,
            "median": 0.004824110815656202,
            "replicas": 16,
            "standard_deviation": 0.004745873627439297,
            "standard_error": 0.0011864684068598242
          },
          "control": "spectral_selector",
          "cumulative_squared_displacement_ratio": {
            "ci95_high": 7.913475792341037,
            "ci95_low": 7.5675824848230935,
            "favorable_count": 16,
            "mean": 7.740529138582065,
            "median": 7.797713292574335,
            "replicas": 16,
            "standard_deviation": 0.32456157194861074,
            "standard_error": 0.08114039298715268
          },
          "nll_auc_gain": {
            "ci95_high": 0.01833207243692278,
            "ci95_low": 0.006902366473027583,
            "favorable_count": 13,
            "mean": 0.012617219454975181,
            "median": 0.013657196779151726,
            "replicas": 16,
            "standard_deviation": 0.010724819630572938,
            "standard_error": 0.0026812049076432344
          },
          "segments": {
            "far_rotation_25_to_30_degrees": {
              "accuracy_gain": {
                "ci95_high": 0.01878804514982298,
                "ci95_low": 0.011274454850177012,
                "favorable_count": 16,
                "mean": 0.015031249999999996,
                "median": 0.014855000000000063,
                "replicas": 16,
                "standard_deviation": 0.00705021642693803,
                "standard_error": 0.0017625541067345075
              },
              "nll_gain": {
                "ci95_high": 0.06863427154536693,
                "ci95_low": 0.041368596196813975,
                "favorable_count": 16,
                "mean": 0.05500143387109045,
                "median": 0.0600542082357407,
                "replicas": 16,
                "standard_deviation": 0.025584162107293085,
                "standard_error": 0.006396040526823271
              }
            },
            "first_ascent": {
              "accuracy_gain": {
                "ci95_high": 0.004631215188671658,
                "ci95_low": -9.277768867163843e-05,
                "favorable_count": 12,
                "mean": 0.00226921875000001,
                "median": 0.0021662499999999807,
                "replicas": 16,
                "standard_deviation": 0.004432657472174555,
                "standard_error": 0.0011081643680436388
              },
              "nll_gain": {
                "ci95_high": 0.015042503410384284,
                "ci95_low": 0.0023581924562601046,
                "favorable_count": 11,
                "mean": 0.008700347933322194,
                "median": 0.00828860296949746,
                "replicas": 16,
                "standard_deviation": 0.011902051334549075,
                "standard_error": 0.0029755128336372686
              }
            },
            "near_upright_0_to_5_degrees": {
              "accuracy_gain": {
                "ci95_high": -0.001063143601796362,
                "ci95_low": -0.004446856398203639,
                "favorable_count": 3,
                "mean": -0.0027550000000000005,
                "median": -0.0026824999999999766,
                "replicas": 16,
                "standard_deviation": 0.0031750343830159416,
                "standard_error": 0.0007937585957539854
              },
              "nll_gain": {
                "ci95_high": -0.002383401573054727,
                "ci95_low": -0.010521337874404827,
                "favorable_count": 1,
                "mean": -0.006452369723729777,
                "median": -0.005165936749875533,
                "replicas": 16,
                "standard_deviation": 0.0076360581166978435,
                "standard_error": 0.0019090145291744609
              }
            },
            "return_leg": {
              "accuracy_gain": {
                "ci95_high": 0.005345567618757567,
                "ci95_low": -0.0004571301187575854,
                "favorable_count": 12,
                "mean": 0.0024442187499999907,
                "median": 0.004134999999999889,
                "replicas": 16,
                "standard_deviation": 0.005444837058990707,
                "standard_error": 0.0013612092647476768
              },
              "nll_gain": {
                "ci95_high": 0.020033172477787277,
                "ci95_low": 0.0033494849833045712,
                "favorable_count": 12,
                "mean": 0.011691328730545923,
                "median": 0.015438892872035481,
                "replicas": 16,
                "standard_deviation": 0.015654780596839946,
                "standard_error": 0.003913695149209986
              }
            },
            "second_ascent": {
              "accuracy_gain": {
                "ci95_high": 0.005549353437415996,
                "ci95_low": 0.0018216831479498405,
                "favorable_count": 13,
                "mean": 0.0036855182926829183,
                "median": 0.0043695121951219384,
                "replicas": 16,
                "standard_deviation": 0.0034977795249551275,
                "standard_error": 0.0008744448812387819
              },
              "nll_gain": {
                "ci95_high": 0.023279966203645957,
                "ci95_low": 0.01286990209854669,
                "favorable_count": 16,
                "mean": 0.018074934151096324,
                "median": 0.01865901401391845,
                "replicas": 16,
                "standard_deviation": 0.009768060545263824,
                "standard_error": 0.002442015136315956
              }
            }
          },
          "worst_panel_nll_auc_gain": {
            "ci95_high": 0.06089212862253861,
            "ci95_low": 0.03922271139442253,
            "favorable_count": 16,
            "mean": 0.05005742000848057,
            "median": 0.050135288543701084,
            "replicas": 16,
            "standard_deviation": 0.02033303323858854,
            "standard_error": 0.005083258309647135
          }
        }
      ],
      "fisher_health": {
        "mean_archive_trace": 219.55238392849546,
        "mean_compression_relative_frobenius_error": 0.512650609144804,
        "mean_kappa": 5.320297081895334,
        "mean_tau": 207.49158619391804
      },
      "optimizer_health": {
        "max_iteration_fraction": 0.0,
        "mean_backtracking_rejections": 0.0,
        "mean_iterations": 11.1171875,
        "mean_relative_final_gradient_norm": 0.001596715489010349
      },
      "q01_mean_cumulative_squared_displacement": 0.05994325072807502,
      "q01_mean_current_accuracy_auc": 0.7716108854166667,
      "q01_mean_current_nll_auc": 0.740554422604603,
      "schedule": "linear"
    },
    {
      "comparisons": [
        {
          "accuracy_auc_gain": {
            "ci95_high": 0.022995626794324747,
            "ci95_low": 0.010862862789008585,
            "favorable_count": 15,
            "mean": 0.016929244791666666,
            "median": 0.015971041666666685,
            "replicas": 16,
            "standard_deviation": 0.011384519075850186,
            "standard_error": 0.0028461297689625466
          },
          "brier_auc_gain": {
            "ci95_high": 0.05198249024827334,
            "ci95_low": 0.03111046476102232,
            "favorable_count": 16,
            "mean": 0.04154647750464783,
            "median": 0.03602046468089909,
            "replicas": 16,
            "standard_deviation": 0.019584817788191086,
            "standard_error": 0.0048962044470477716
          },
          "control": "gauge_no_ridge",
          "cumulative_squared_displacement_ratio": {
            "ci95_high": 0.0034882846255314954,
            "ci95_low": 0.0023941376703954776,
            "favorable_count": 16,
            "mean": 0.0029412111479634865,
            "median": 0.0026391123934456765,
            "replicas": 16,
            "standard_deviation": 0.001026669345671889,
            "standard_error": 0.00025666733641797223
          },
          "nll_auc_gain": {
            "ci95_high": 0.24187695735812964,
            "ci95_low": 0.15742112385845275,
            "favorable_count": 16,
            "mean": 0.1996490406082912,
            "median": 0.18171901734555768,
            "replicas": 16,
            "standard_deviation": 0.07924732131298419,
            "standard_error": 0.019811830328246047
          },
          "segments": {
            "far_rotation_25_to_30_degrees": {
              "accuracy_gain": {
                "ci95_high": -0.02979794160366412,
                "ci95_low": -0.05360896629107272,
                "favorable_count": 0,
                "mean": -0.04170345394736842,
                "median": -0.03398157894736836,
                "replicas": 16,
                "standard_deviation": 0.0223425646992364,
                "standard_error": 0.0055856411748091
              },
              "nll_gain": {
                "ci95_high": 0.21739899967911938,
                "ci95_low": 0.0496765500938898,
                "favorable_count": 13,
                "mean": 0.1335377748865046,
                "median": 0.0949227822868447,
                "replicas": 16,
                "standard_deviation": 0.15737876595264824,
                "standard_error": 0.03934469148816206
              }
            },
            "first_ascent": {
              "accuracy_gain": {
                "ci95_high": 0.04685030546399032,
                "ci95_low": 0.026583757036009646,
                "favorable_count": 16,
                "mean": 0.03671703124999998,
                "median": 0.03579500000000002,
                "replicas": 16,
                "standard_deviation": 0.019016681366166178,
                "standard_error": 0.0047541703415415444
              },
              "nll_gain": {
                "ci95_high": 0.34110433360760684,
                "ci95_low": 0.22383760489298377,
                "favorable_count": 16,
                "mean": 0.2824709692502953,
                "median": 0.24766902462482443,
                "replicas": 16,
                "standard_deviation": 0.11003472163714818,
                "standard_error": 0.027508680409287046
              }
            },
            "near_upright_0_to_5_degrees": {
              "accuracy_gain": {
                "ci95_high": 0.06696556838212836,
                "ci95_low": 0.0496054842494506,
                "favorable_count": 16,
                "mean": 0.058285526315789475,
                "median": 0.05210131578947369,
                "replicas": 16,
                "standard_deviation": 0.016289462885805465,
                "standard_error": 0.004072365721451366
              },
              "nll_gain": {
                "ci95_high": 0.2674299596371169,
                "ci95_low": 0.20036307202748754,
                "favorable_count": 16,
                "mean": 0.23389651583230223,
                "median": 0.203747202156092,
                "replicas": 16,
                "standard_deviation": 0.0629307766157139,
                "standard_error": 0.015732694153928476
              }
            },
            "return_leg": {
              "accuracy_gain": {
                "ci95_high": 0.01735389116385571,
                "ci95_low": 0.0022092338361443167,
                "favorable_count": 12,
                "mean": 0.009781562500000014,
                "median": 0.010418750000000032,
                "replicas": 16,
                "standard_deviation": 0.014210664624235541,
                "standard_error": 0.0035526661560588853
              },
              "nll_gain": {
                "ci95_high": 0.22430694315906435,
                "ci95_low": 0.10053267793404125,
                "favorable_count": 16,
                "mean": 0.1624198105465528,
                "median": 0.1392454041862488,
                "replicas": 16,
                "standard_deviation": 0.11614092905262083,
                "standard_error": 0.029035232263155207
              }
            },
            "second_ascent": {
              "accuracy_gain": {
                "ci95_high": 0.010530185305662616,
                "ci95_low": -0.0038432950617601474,
                "favorable_count": 10,
                "mean": 0.0033434451219512343,
                "median": 0.002386585365853733,
                "replicas": 16,
                "standard_deviation": 0.013487047251358658,
                "standard_error": 0.0033717618128396645
              },
              "nll_gain": {
                "ci95_high": 0.1950878762201011,
                "ci95_low": 0.1060258484802059,
                "favorable_count": 15,
                "mean": 0.1505568623501535,
                "median": 0.13347737753565725,
                "replicas": 16,
                "standard_deviation": 0.08356944495866456,
                "standard_error": 0.02089236123966614
              }
            }
          },
          "worst_panel_nll_auc_gain": {
            "ci95_high": 0.29384156341208467,
            "ci95_low": 0.15757358863851975,
            "favorable_count": 15,
            "mean": 0.22570757602530223,
            "median": 0.20994627859155324,
            "replicas": 16,
            "standard_deviation": 0.12786413364319751,
            "standard_error": 0.03196603341079938
          }
        },
        {
          "accuracy_auc_gain": {
            "ci95_high": 0.005133822061435492,
            "ci95_low": 0.002554198771897789,
            "favorable_count": 15,
            "mean": 0.0038440104166666406,
            "median": 0.004046041666666611,
            "replicas": 16,
            "standard_deviation": 0.002420534227434197,
            "standard_error": 0.0006051335568585493
          },
          "brier_auc_gain": {
            "ci95_high": 0.006777000777879058,
            "ci95_low": 0.003591137265962475,
            "favorable_count": 15,
            "mean": 0.005184069021920767,
            "median": 0.00594295955323082,
            "replicas": 16,
            "standard_deviation": 0.0029893867472059418,
            "standard_error": 0.0007473466868014854
          },
          "control": "spectral_selector",
          "cumulative_squared_displacement_ratio": {
            "ci95_high": 7.827162434249132,
            "ci95_low": 7.431963831461827,
            "favorable_count": 16,
            "mean": 7.62956313285548,
            "median": 7.640114850502906,
            "replicas": 16,
            "standard_deviation": 0.37082613905702366,
            "standard_error": 0.09270653476425592
          },
          "nll_auc_gain": {
            "ci95_high": 0.02345851529319251,
            "ci95_low": 0.013012391263006502,
            "favorable_count": 16,
            "mean": 0.018235453278099507,
            "median": 0.019273253909349586,
            "replicas": 16,
            "standard_deviation": 0.009801896603135213,
            "standard_error": 0.002450474150783803
          },
          "segments": {
            "far_rotation_25_to_30_degrees": {
              "accuracy_gain": {
                "ci95_high": 0.017677662118819663,
                "ci95_low": 0.011669048407496104,
                "favorable_count": 16,
                "mean": 0.014673355263157883,
                "median": 0.014759210526315791,
                "replicas": 16,
                "standard_deviation": 0.005638053899837271,
                "standard_error": 0.0014095134749593178
              },
              "nll_gain": {
                "ci95_high": 0.07173468703310742,
                "ci95_low": 0.04567432125276638,
                "favorable_count": 16,
                "mean": 0.0587045041429369,
                "median": 0.06414754773441123,
                "replicas": 16,
                "standard_deviation": 0.024453185706070683,
                "standard_error": 0.006113296426517671
              }
            },
            "first_ascent": {
              "accuracy_gain": {
                "ci95_high": 0.004465920853696035,
                "ci95_low": 0.0013672041463038903,
                "favorable_count": 13,
                "mean": 0.0029165624999999626,
                "median": 0.0024387499999999895,
                "replicas": 16,
                "standard_deviation": 0.0029076144109045723,
                "standard_error": 0.0007269036027261431
              },
              "nll_gain": {
                "ci95_high": 0.018844406022280505,
                "ci95_low": 0.007522365951687011,
                "favorable_count": 15,
                "mean": 0.013183385986983757,
                "median": 0.010229425647556745,
                "replicas": 16,
                "standard_deviation": 0.010623793647081077,
                "standard_error": 0.0026559484117702693
              }
            },
            "near_upright_0_to_5_degrees": {
              "accuracy_gain": {
                "ci95_high": -0.0013282984959516037,
                "ci95_low": -0.004149004135627369,
                "favorable_count": 1,
                "mean": -0.0027386513157894865,
                "median": -0.0028868421052631876,
                "replicas": 16,
                "standard_deviation": 0.002646748683826407,
                "standard_error": 0.0006616871709566017
              },
              "nll_gain": {
                "ci95_high": -0.003318194531493063,
                "ci95_low": -0.009315186294285299,
                "favorable_count": 2,
                "mean": -0.006316690412889181,
                "median": -0.008045303862973252,
                "replicas": 16,
                "standard_deviation": 0.00562714869351368,
                "standard_error": 0.00140678717337842
              }
            },
            "return_leg": {
              "accuracy_gain": {
                "ci95_high": 0.0055027279562757025,
                "ci95_low": 0.0019835220437242973,
                "favorable_count": 14,
                "mean": 0.003743125,
                "median": 0.004771250000000005,
                "replicas": 16,
                "standard_deviation": 0.0033021714446709225,
                "standard_error": 0.0008255428611677306
              },
              "nll_gain": {
                "ci95_high": 0.02559239934193555,
                "ci95_low": 0.010343285565389035,
                "favorable_count": 14,
                "mean": 0.017967842453662293,
                "median": 0.018042442753464005,
                "replicas": 16,
                "standard_deviation": 0.01430867909429677,
                "standard_error": 0.0035771697735741924
              }
            },
            "second_ascent": {
              "accuracy_gain": {
                "ci95_high": 0.006935811849482424,
                "ci95_low": 0.00308796863832248,
                "favorable_count": 14,
                "mean": 0.005011890243902452,
                "median": 0.004935365853658558,
                "replicas": 16,
                "standard_deviation": 0.003610541210488953,
                "standard_error": 0.0009026353026222383
              },
              "nll_gain": {
                "ci95_high": 0.03089987903794602,
                "ci95_low": 0.017291068585822945,
                "favorable_count": 15,
                "mean": 0.024095473811884482,
                "median": 0.025870552715441086,
                "replicas": 16,
                "standard_deviation": 0.012769535624688624,
                "standard_error": 0.003192383906172156
              }
            }
          },
          "worst_panel_nll_auc_gain": {
            "ci95_high": 0.05675936745160623,
            "ci95_low": 0.03827553629816771,
            "favorable_count": 16,
            "mean": 0.04751745187488697,
            "median": 0.04390688740034887,
            "replicas": 16,
            "standard_deviation": 0.017343906818669826,
            "standard_error": 0.004335976704667456
          }
        }
      ],
      "fisher_health": {
        "mean_archive_trace": 231.4251060544724,
        "mean_compression_relative_frobenius_error": 0.5205447850038418,
        "mean_kappa": 5.6066653630706105,
        "mean_tau": 218.65994915975384
      },
      "optimizer_health": {
        "max_iteration_fraction": 0.0,
        "mean_backtracking_rejections": 0.0,
        "mean_iterations": 11.111979166666666,
        "mean_relative_final_gradient_norm": 0.0016149827780033614
      },
      "q01_mean_cumulative_squared_displacement": 0.06024253825564338,
      "q01_mean_current_accuracy_auc": 0.7513709114583333,
      "q01_mean_current_nll_auc": 0.8123735014919689,
      "schedule": "sigmoid"
    }
  ],
  "status": "complete_exploratory_probe"
}
```
