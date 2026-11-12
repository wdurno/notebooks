"""Tiny CPU tensors and real sealed RunStores; no encoder or simulator runs."""

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from numerical_experiment import picar_feature_health as health
from numerical_experiment.picar.models import batch_digest
from numerical_experiment.picar.report import ArtifactError, read_completed_unit
from numerical_experiment.picar.storage import IntegrityError, RunCollisionError, RunStore


@pytest.fixture
def tensors():
    history = {"images": torch.zeros(1, 2, 3, 2, 2),
        "executed_actions": torch.tensor([[[90., 80., 70., 60.], [.1, .2, .3, .4]]]),
        "elapsed": torch.tensor([[[99.], [.5]]]), "start_time": torch.tensor([0.]),
        "decision_time": torch.tensor([[0., 1.]]), "episode_deadline": torch.tensor([10.]),
        "contexts": [[{"text": "Drive toward the target.", "issued_at": 0.}]]}
    parameters = {"projection.ltm.weight": torch.full((2, 7), .1),
        "projection.ltm.bias": torch.tensor([.1, -.2]),
        "projection.stm.A": torch.full((1, 7), .2),
        "projection.stm.B": torch.tensor([[.3], [-.4]])}
    state = {"learner": {"parameters": parameters, "version": 7},
             "eligible": [{"arrival": 11, "history": history}]}
    rows = [{"arrival": 11, "condition": condition, "source_history_digest": batch_digest(history),
             "features": [float(i + 1), float(2 * (i + 1))], "logits": [i * 1e-7, 1.]}
            for i, condition in enumerate(health.CONDITIONS)]
    return state, {"model_version": 7, "rows": rows}


def sources(tmp_path, tensors, *, policy="journal-v1", initial_kind="phase2_initial_asset"):
    state, language = tensors
    paths = [tmp_path / "initial", tmp_path / "language"]
    for path, kind, payload, summary in (
        (paths[0], initial_kind, state, {}),
        (paths[1], "phase2_language", {"stage": "done"}, language)):
        with RunStore(path, {"schema_version": 1, "fixture": kind}, checkpoint_policy=policy) as store:
            store.checkpoint(payload, progress={"phase": 2, "status": "done"})
            store.complete({"schema_version": 1, "phase": 2, "kind": kind, **summary})
    return paths


def contents(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("normalization", health.NORMALIZATIONS)
def test_reconstruction_uses_effective_weight_bias_last_controls_and_explicit_rms(tensors, normalization):
    state, language = tensors
    result = health.analyze_projection(state, language, normalization=normalization)
    p = state["learner"]["parameters"]
    features = torch.tensor([r["features"] for r in language["rows"]])
    if normalization != "none":
        features = F.rms_norm(features, (features.shape[-1],), eps=1e-6)
    history = state["eligible"][0]["history"]
    control = torch.cat((history["executed_actions"][0, -1], history["elapsed"][0, -1]))
    x = torch.cat((features, control.expand(4, -1)), dim=-1)
    z = F.linear(x, p["projection.ltm.weight"] + p["projection.stm.B"] @ p["projection.stm.A"], p["projection.ltm.bias"])
    assert result["raw_projection_abs_median"] == float(z.abs().median())
    if normalization == "feature_and_projection_rms":
        z = F.rms_norm(z, (z.shape[-1],), eps=1e-6)
    slope = 1 - z.tanh().square()
    assert result["abs_preactivation_median"] == float(z.abs().median())
    assert result["derivative_median"] == float(slope.median())
    assert result["derivative_max"] == float(slope.max())
    assert result["saturated_fraction"] == float((z.tanh().abs() > .99).double().mean())
    assert result["normalization"] == normalization and result["eps"] == 1e-6
    assert result["device"] == "cpu" and result["dtype"] == "torch.float32"
    assert all(sum(h["counts"]) == 8 for h in result["distributions"].values())


def test_deltas_preserve_original_digest_and_explicit_norms(tensors):
    state, language = tensors
    result = health.analyze_projection(state, language)
    assert len(result["instruction_deltas"]) == 6
    first = result["instruction_deltas"][0]
    assert first["instruction_a"] == "forward" and first["instruction_b"] == "left"
    assert first["feature_l2"] == pytest.approx(5 ** .5)
    assert first["feature_linf"] == 2.
    assert result["feature_delta_linf_per_arrival_max_range"] == [6., 6.]
    assert result["feature_delta_linf_from_forward_max_range"] == [6., 6.]
    assert first["logit_linf"] == pytest.approx(1e-7)
    assert all(r["source_history_digest"] == language["rows"][0]["source_history_digest"] for r in result["original_digests"])


def test_complete_saturation_and_local_slope_are_not_replaced_by_a_stable_surrogate(tensors):
    state, language = tensors
    state["learner"]["parameters"]["projection.ltm.bias"].fill_(20.)
    result = health.analyze_projection(state, language)
    assert result["saturated_fraction"] == 1.
    assert result["derivative_median"] == result["derivative_max"] == 0.


@pytest.mark.parametrize("change", ["unknown_arrival", "duplicate", "missing", "digest", "version", "nonfinite", "scale"])
def test_incompatible_or_unpaired_evidence_fails_without_repair(tensors, change):
    state, language = tensors
    if change == "unknown_arrival":
        language["rows"][0]["arrival"] = 12
    elif change == "duplicate":
        language["rows"][1]["condition"] = "forward"
    elif change == "missing":
        language["rows"].pop()
    elif change == "digest":
        language["rows"][0]["source_history_digest"] = "wrong"
    elif change == "version":
        language["model_version"] = 8
    elif change == "nonfinite":
        language["rows"][0]["features"][0] = float("nan")
    else:
        state["learner"]["parameters"]["projection.stm.scale"] = torch.tensor(2.)
    with pytest.raises(ValueError):
        health.analyze_projection(state, language)


@pytest.mark.parametrize("normalization,eps", [("unknown", 1e-6), ("none", 0.), ("feature_rms", float("nan")), ("feature_rms", True)])
def test_invalid_normalization_is_rejected_before_any_io(normalization, eps):
    with pytest.raises(ValueError, match="normalization or epsilon"):
        health.run("unused-initial", "unused-language", "unused-output", normalization=normalization, eps=eps)


@pytest.mark.parametrize("policy", ["legacy", "journal-v1"])
@pytest.mark.parametrize("initial_kind", ["phase2_initial_asset", "phase2_bootstrap"])
def test_run_verifies_sources_cpu_only_immutable_resume_and_report_schema(tmp_path, tensors, monkeypatch, policy, initial_kind):
    initial, language = sources(tmp_path, tensors, policy=policy, initial_kind=initial_kind)
    before = [contents(p) for p in (initial, language)]
    loaded = []
    original = torch.load

    def load(*args, **kwargs):
        loaded.append(kwargs)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU diagnostic touched CUDA")

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    output = tmp_path / "health"
    result = health.run(initial, language, output)
    assert result["kind"] == "phase2_feature_health"
    assert loaded == [{"map_location": "cpu", "weights_only": False}]
    saved = read_completed_unit(output)
    assert saved["config"]["schema_version"] == 1
    assert saved["summary"] == result
    assert saved["manifest"]["runtime"]["git_commit"]
    assert result["source_sha256"]["numerical_experiment/picar_feature_health.py"]
    assert result["sources"]["initial_asset"]["kind"] == initial_kind
    assert result["mathematical_definition"]["stm_scale"] == 1.
    sealed = contents(output)
    assert health.run(initial, language, output, resume=True) == result
    assert contents(output) == sealed
    assert [contents(p) for p in (initial, language)] == before
    with pytest.raises(RunCollisionError):
        health.run(initial, language, output)
    with pytest.raises(RunCollisionError, match="configuration"):
        health.run(initial, language, output, resume=True, normalization="feature_rms")


def test_initial_allowed_kind_tuple_rejects_other_units_and_keeps_sources_sealed(tmp_path, tensors):
    initial, language = sources(tmp_path, tensors, initial_kind="phase2_teacher_readiness")
    with pytest.raises(ValueError, match="phase2_bootstrap"):
        health.load_inputs(initial, language)


def test_output_cannot_be_nested_under_either_sealed_source(tmp_path, tensors):
    initial, language = sources(tmp_path, tensors, initial_kind="phase2_bootstrap")
    for source in (initial, language):
        before = contents(source)
        with pytest.raises(RunCollisionError, match="completed artifact"):
            health.run(initial, language, source / "health")
        assert contents(source) == before
        assert not (source / "health").exists()


def test_corrupt_language_seal_fails_before_unpickling_or_creating_output(tmp_path, tensors, monkeypatch):
    initial, language = sources(tmp_path, tensors)
    (language / "summary.json").write_text('{"kind":"phase2_language","rows":[]}')

    def forbidden(*args, **kwargs):
        raise AssertionError("unpickled before verifying both sources")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ArtifactError, match="hash/size"):
        health.run(initial, language, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_missing_completion_and_corrupt_checkpoint_do_not_repair_sources(tmp_path, tensors):
    initial, language = sources(tmp_path, tensors)
    manifest = json.loads((initial / "manifest.json").read_text())
    payload = initial / "transactions" / manifest["current"]["commit_id"] / "state.pt"
    payload.write_bytes(b"bad checkpoint")
    before = contents(initial)
    with pytest.raises(IntegrityError, match="checksum"):
        health.run(initial, language, tmp_path / "output")
    assert contents(initial) == before and not (tmp_path / "output").exists()
    (language / "COMPLETED").unlink()
    with pytest.raises(ArtifactError, match="incomplete"):
        with health._completed(language, "phase2_language"):
            pass


def test_cli_accepts_normalization_and_emits_only_four_numbers(tmp_path, tensors, capsys):
    initial, language = sources(tmp_path, tensors)
    health.main(["--initial-asset", str(initial), "--language-unit", str(language),
                 "--output", str(tmp_path / "output"), "--normalization", "feature_rms", "--eps", "0.0001"])
    result = json.loads(capsys.readouterr().out)
    assert len(result) == 4 and all(isinstance(v, float) for v in result.values())
    config = read_completed_unit(tmp_path / "output")["config"]
    assert config["normalization"] == "feature_rms" and config["eps"] == .0001
