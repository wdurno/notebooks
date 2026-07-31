import json
from pathlib import Path

from src.config import load_config
from src.initialization import replica_bundle_id


REPO_ROOT = Path(__file__).parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "mnist_experiment"
MANIFEST = EXPERIMENT_ROOT / "phase5_pilot_manifest.json"


def test_phase5_manifest_describes_the_bounded_pilot_exactly() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    center = load_config(EXPERIMENT_ROOT / manifest["center"]["config"])
    new_configs = [
        load_config(EXPERIMENT_ROOT / cell["config"])
        for cell in manifest["new_cells"]
    ]
    all_configs = [center, *new_configs]

    assert manifest["schema_version"] == 1
    assert len(new_configs) == 7
    assert len({config.run_id for config in all_configs}) == 8
    assert all(config.schema_version == 5 for config in all_configs)
    assert all(
        config.experiment.startswith("mnist_lfu_phase5_")
        for config in new_configs
    )

    factorial = {
        (config.data.num_p_steps, config.data.samples_per_step)
        for config, cell in zip(new_configs, manifest["new_cells"], strict=True)
        if "factorial" in cell["roles"]
    }
    factorial.add((center.data.num_p_steps, center.data.samples_per_step))
    assert factorial == {(9, 64), (9, 128), (21, 64), (21, 128)}

    matched = [
        config.data.num_p_steps * config.data.samples_per_step
        for config, cell in zip(new_configs, manifest["new_cells"], strict=True)
        if "compute_matched" in cell["roles"]
    ]
    assert matched == [1152, 1344]

    ema_axis = {
        center.estimator.ema_gain,
        *(
            config.estimator.ema_gain
            for config, cell in zip(
                new_configs,
                manifest["new_cells"],
                strict=True,
            )
            if "ema_gain_axis" in cell["roles"]
        ),
    }
    assert ema_axis == {0.1, 0.25, 0.5}

    half_life_axis = {
        center.estimator.ridge_half_life_steps,
        *(
            config.estimator.ridge_half_life_steps
            for config, cell in zip(
                new_configs,
                manifest["new_cells"],
                strict=True,
            )
            if "ridge_half_life_axis" in cell["roles"]
        ),
    }
    assert half_life_axis == {4.0, 8.0, 16.0}
    assert {
        config.estimator.ridge_amplitude_epsilon for config in all_configs
    } == {1e-6}
    assert {
        config.estimator.ridge_coherence_threshold for config in all_configs
    } == {0.75}


def test_phase5_treatment_axes_reuse_the_center_replica_bundle() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    center = load_config(EXPERIMENT_ROOT / manifest["center"]["config"])
    center_bundle = replica_bundle_id(center)

    for cell in manifest["new_cells"]:
        if not {
            "ema_gain_axis",
            "ridge_half_life_axis",
        }.intersection(cell["roles"]):
            continue
        config = load_config(EXPERIMENT_ROOT / cell["config"])
        assert replica_bundle_id(config) == center_bundle
