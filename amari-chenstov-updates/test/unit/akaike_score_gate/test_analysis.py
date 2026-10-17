"""Scientific scalar summaries and artifact contracts; no experiment execution."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from mnist_experiment.akaike_score_gate import analysis


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def make_root(path):
    # study.json is a raw resolved config, not a Study/UnitStore serialization.
    write_json(path / "study.json", {"schema_version": 1, "seed_root": 14, "smoke": True})
    write_json(path / "progress.json", {"status": "partial", "phases": {"phase0": {"planned": 2}}})
    write_json(path / "lab_log.json", [{"message": "Synthetic report fixture; no training."}])
    return path


def seal_unit(root, *, phase="phase4", kind="trajectory", index=0, policy="no_update",
              environment="mixture_m18", metrics=None, summary=None, detail=None, audit=None):
    path = root / phase / f"{kind}-{index}-{environment}-{policy}"
    config = {"schema_version": 1, "phase": phase, "kind": kind, "index": index,
              "environment": environment, "policy": policy, "detail": detail or {}}
    write_json(path / "config.json", config)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(path / "manifest.json", {"manifest_schema_version": 1, "artifact_schema_version": 1,
               "metric_schema_version": 1, "config_hash": digest, "status": "completed"})
    write_json(path / "summary.json", summary if summary is not None else {
        "phase": phase, "kind": kind, "environment": environment, "replica_index": index,
        "policy": policy, "expected_nines": 9.9, "total_observations": 180,
        "nll_auc": .8 + index / 100, "retention_nll": .2, "online_seconds": 3 + index,
    })
    if kind == "trajectory" or metrics is not None:
        write_json(path / "metrics.json", trajectory_rows() if metrics is None else metrics)
    if audit is not None:
        write_json(path / "audit_metrics.json", audit)
    reseal(path)
    (path / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return path


def reseal(path):
    write_json(path / "integrity.json", {
        name.name: hashlib.sha256(name.read_bytes()).hexdigest()
        for name in path.glob("*.json") if name.name not in {"integrity.json", "manifest.json"}
    })


def trajectory_rows():
    return [{"step": step - 1, "observations": observations, "expected_nines": expected,
             "realized_nines": realized, "current_nll": nll, "retention_nll": .2,
             "fixed_nll": nll, "current_accuracy": .7, "retention_accuracy": .9,
             "nine_precision": None if step == 0 else .5,
             "nine_precision_defined": step != 0, "nine_recall": .5 if step else 0.,
             "nine_false_positive_rate": .05 if step else 0.,
             "online_seconds": step, "charged_seconds": step * 2,
             "selected_depth": step, "active_parameters": step * 10, "n_eff": step * 4,
             "d_eff": None if step == 0 else 2., "gate_status": "baseline" if not step else "closed"}
            for step, (observations, expected, realized, nll) in enumerate(
                [(0, 0., 0, 1.2), (60, 2., 1, .8), (120, 5., 5, .7), (180, 9.9, 10, .6)])]


def test_auc_is_trapezoidal_and_exposure_normalized():
    assert analysis.exposure_normalized_auc([0, 2, 10], [3, 1, 2], origin=0) == pytest.approx(1.6)
    assert analysis.exposure_normalized_auc([8, 10, 18], [3, 1, 2], origin=8) == pytest.approx(1.6)
    assert analysis.exposure_normalized_auc([0, 2, 10], [3, 1, 2], origin=0, stop=6) == pytest.approx(1.5)


@pytest.mark.parametrize("x,y,origin,stop", [
    ([2, 10], [1, 2], 0, None), ([0, 0], [1, 2], 0, None),
    ([0, 10], [1, float("nan")], 0, None), ([0, 10], [1, 2], 0, 11),
    ([0, 10], [1, 2], 0, 0), ([0], [1], 0, None),
])
def test_auc_refuses_imputation_and_extrapolation(x, y, origin, stop):
    with pytest.raises(ValueError):
        analysis.exposure_normalized_auc(x, y, origin=origin, stop=stop)


def test_early_auc_uses_first_complete_arrival_and_reports_actual_count():
    result = analysis.early_exposure_summary(trajectory_rows(), origin=0, total_observations=180)
    assert [row["observations"] for row in result] == [60, 120, 180]
    assert result[0]["nll_auc"] == pytest.approx(1.)
    assert result[1]["nll_auc"] == pytest.approx(.875)
    missing_origin = analysis.early_exposure_summary(trajectory_rows()[1:], origin=0)
    assert missing_origin[-1]["nll_auc"] is None
    assert "origin" in missing_origin[-1]["missing_reason"]


def test_pairing_uses_replica_ids_not_row_order_and_bootstraps_differences():
    rows = [{"phase": "phase5", "environment": "rotation_linear", "policy": policy,
             "replica_index": replica, "nll_auc": value}
            for policy, replica, value in [("asg", 1, 10), ("fixed", 2, 101),
                                          ("asg", 2, 100), ("fixed", 1, 11), ("asg", 9, 0)]]
    result = analysis.paired_effects(rows, "asg", "fixed", "nll_auc", seed=19)[0]
    assert result["differences"] == [-1, -1]
    assert result["mean"] == result["ci95_low"] == result["ci95_high"] == -1
    assert result["unpaired_treatment"] == 1
    assert result == analysis.paired_effects(list(reversed(rows)), "asg", "fixed", "nll_auc", seed=19)[0]


def test_pairing_never_crosses_schedule_scale_budget_or_phase():
    rows = []
    for phase in ("phase4", "phase5"):
        for schedule in ("linear", "sigmoid"):
            for scale in (4.4, 9.9, 100.1):
                for budget in ("strict", "principal"):
                    for policy in ("asg", "fixed"):
                        rows.append(dict(phase=phase, schedule=schedule, expected_nines=scale,
                                         budget_tier=budget, policy=policy, replica_index=0, nll_auc=1.))
    assert len(analysis.paired_effects(rows, "asg", "fixed", "nll_auc")) == 24
    with pytest.raises(ValueError, match="Duplicate replica"):
        analysis.paired_effects(rows + [rows[0]], "asg", "fixed", "nll_auc")


def test_missing_paired_endpoint_is_counted_not_zero_filled():
    rows = [{"policy": policy, "replica_index": i, "nll_auc": value}
            for policy, i, value in [("a", 1, None), ("b", 1, 8), ("a", 2, 2), ("b", 2, 3)]]
    result = analysis.paired_effects(rows, "a", "b", "nll_auc")[0]
    assert result["n"] == 1 and result["missing"] == 1
    assert result["mean"] == -1 and result["ci95_low"] is None


def test_seeded_bootstrap_is_reproducible_and_one_replica_has_no_ci():
    assert analysis.mean_interval([1., 2., 8.], seed=7) == analysis.mean_interval([1., 2., 8.], seed=7)
    assert analysis.mean_interval([1.])["ci95_high"] is None
    assert analysis.mean_interval([None])["mean"] is None


def test_undefined_precision_and_absent_flag_never_become_zero(tmp_path):
    root = make_root(tmp_path)
    for i, flag, precision in [(0, True, .5), (1, False, 0.), (2, None, .9)]:
        seal_unit(root, index=i, metrics=[{"step": 0, "observations": 0, "nine_precision": precision,
                                        "nine_precision_defined": flag}])
    units = analysis.trajectory_groups(analysis.load_report(root))[0]
    result = analysis.pointwise_summary(units, "nine_precision")[0]
    assert result["mean"] == .5 and result["n"] == 1
    assert result["defined_fraction"] == pytest.approx(1 / 3)
    assert result["missing"] == 2
    assert analysis.metric_value({"nine_precision_defined": False, "nine_recall": 0,
                                  "nine_false_positive_rate": 0}, "standardized_nine_precision") is None
    assert analysis.metric_value({"nine_precision_defined": True, "nine_recall": .5,
                                  "nine_false_positive_rate": .05}, "standardized_nine_precision") == pytest.approx(.05 / .095)


def test_targets_require_two_consecutive_evaluations_and_second_cost():
    result = analysis.target_attainment(trajectory_rows(), .8, origin=0)
    assert result["reached"] and not result["censored"]
    assert result["observations"] == 120 and result["online_seconds"] == 2
    assert result["baseline_satisfied"] is False


def test_targets_keep_censored_replicas_and_flag_baseline_and_later_loss():
    rows = trajectory_rows()
    for row, nll in zip(rows, [.7, .6, 1.2, .7]):
        row["fixed_nll"] = nll
    reached = analysis.target_attainment(rows, .8, origin=0)
    censored = analysis.target_attainment(rows, .5, origin=0)
    assert reached["baseline_satisfied"] and reached["lost_after_attainment"]
    assert reached["online_seconds"] == 1
    assert censored["status"] == "right_censored" and censored["observations"] == 180
    result = analysis.target_summary([reached, censored])
    assert result["attainment_fraction"] == .5 and result["replicas"] == 2 and result["censored"] == 1
    assert not any(key.startswith("mean") for key in result)


def test_missing_target_evaluation_breaks_pair_and_current_nll_is_not_substituted():
    rows = trajectory_rows()
    for row, value in zip(rows, [1., .7, None, .7]):
        row["fixed_nll"] = value
        row["current_nll"] = .1
    result = analysis.target_attainment(rows, .8, origin=0)
    assert result["censored"] and result["status"] == "missing_metrics"
    assert result["missing_evaluations"] == 1
    assert analysis.target_attainment(rows[1:], .8, origin=0)["baseline_satisfied"] is None


def test_loader_phase_order_and_incomplete_units_are_never_read(tmp_path):
    root = make_root(tmp_path)
    for phase in reversed(list(analysis.PHASE_ORDER)):
        seal_unit(root, phase=phase, kind="contracts", summary={"ok": True})
    incomplete = root / "phase0" / ".incomplete" / "broken"
    incomplete.mkdir(parents=True)
    (incomplete / "summary.json").write_text("not json", encoding="utf-8")
    (root / "phase3" / "unsealed").mkdir()
    result = analysis.load_report(root)
    assert [unit["config"]["phase"] for unit in result["units"]] == list(analysis.PHASE_ORDER)
    assert result["incomplete"] == 2


@pytest.mark.parametrize("mutation", ["checksum", "schema", "missing_metrics", "uncovered", "config", "escape"])
def test_loader_rejects_corrupt_or_incompatible_sealed_artifacts(tmp_path, mutation):
    root = make_root(tmp_path)
    path = seal_unit(root)
    if mutation == "checksum":
        write_json(path / "summary.json", {"tampered": True})
    elif mutation == "schema":
        value = json.loads((path / "manifest.json").read_text())
        value["metric_schema_version"] = 2
        write_json(path / "manifest.json", value)
    elif mutation == "missing_metrics":
        (path / "metrics.json").unlink()
    elif mutation == "uncovered":
        value = json.loads((path / "integrity.json").read_text())
        del value["metrics.json"]
        write_json(path / "integrity.json", value)
    elif mutation == "config":
        value = json.loads((path / "config.json").read_text())
        value["index"] = 9
        write_json(path / "config.json", value)
        reseal(path)
    elif mutation == "escape":
        value = json.loads((path / "integrity.json").read_text())
        value["../outside.json"] = "0" * 64
        write_json(path / "integrity.json", value)
    with pytest.raises(analysis.ArtifactError):
        analysis.load_report(root)


def test_source_metrics_are_not_treated_as_trajectories(tmp_path):
    root = make_root(tmp_path)
    seal_unit(root, phase="phase1", kind="source", environment="mixture", policy=None,
              summary={"family": "mixture", "source_nll": .3},
              metrics=[{"epoch": 1, "validation_nll": .3}])
    report = analysis.load_report(root)
    assert len(report["units"]) == 1
    assert analysis.trajectory_groups(report) == []


def test_rotation_composition_does_not_split_replica_group(tmp_path):
    root = make_root(tmp_path)
    for i in range(2):
        seal_unit(root, environment="rotation_linear", index=i,
                  summary={"family": "rotation", "expected_nines": 40 + i,
                           "total_observations": 480, "nll_auc": 1.})
    report = analysis.load_report(root)
    assert len(analysis.trajectory_groups(report)) == 1
    assert analysis.pointwise_summary(analysis.trajectory_groups(report)[0], "current_nll")[0]["n"] == 2


def test_policy_evidence_protocol_and_retention_are_not_pooled(tmp_path):
    root = make_root(tmp_path)
    for i, half_life in enumerate([4., 8.]):
        seal_unit(root, index=i, detail={"policy": {"half_life": half_life, "cadence": 1},
                                       "episode_arrivals": 32, "retention": {"omega": 19}})
    assert len(analysis.trajectory_groups(analysis.load_report(root))) == 2


def test_report_requires_study_but_explicitly_allows_missing_progress_log(tmp_path):
    with pytest.raises(analysis.ArtifactError, match="study.json"):
        analysis.load_report(tmp_path)
    write_json(tmp_path / "study.json", {"schema_version": 1})
    report = analysis.load_report(tmp_path)
    assert report["progress"] is None and report["lab_log"] is None


def test_scalar_pr_schema_undefined_endpoints_and_missing_replicas(tmp_path):
    root = make_root(tmp_path)
    for index in range(3):
        path = seal_unit(root, index=index)
        if index < 2:
            write_json(path / "pr_curves.json", [
                {"step": 2, "observations": 180, "threshold": .2, "precision": .3 + .2 * index,
                 "recall": .9, "precision_defined": True, "false_positive_rate": .1},
                {"step": 2, "observations": 180, "threshold": .8, "precision": .9,
                 "recall": .2, "precision_defined": True, "false_positive_rate": .01},
                {"step": 2, "observations": 180, "threshold": 1., "precision": None,
                 "recall": 0., "precision_defined": False, "false_positive_rate": 0.},
            ])
            reseal(path)
    units = analysis.trajectory_groups(analysis.load_report(root))[0]
    rows = analysis.pr_curve_summary(units, points=21)
    middle = rows[11]
    assert middle["mean"] == pytest.approx(.65)
    assert middle["recall"] == pytest.approx(.55)
    assert middle["n"] == 2 and middle["n_total"] == 3
    assert middle["defined_fraction"] == pytest.approx(2 / 3)
    assert rows[0]["mean"] is None and rows[-1]["mean"] is None


def test_unknown_phase_and_duplicate_exposure_fail_clearly(tmp_path):
    root = make_root(tmp_path)
    path = root / "phase10"
    path.mkdir()
    with pytest.raises(analysis.ArtifactError, match="Unknown phase"):
        analysis.load_report(root)
    path.rmdir()
    rows = trajectory_rows()
    rows[1]["observations"] = 0
    seal_unit(root, metrics=rows)
    with pytest.raises(analysis.ArtifactError, match="Duplicate evaluation exposure"):
        analysis.load_report(root)


@pytest.mark.parametrize("setting,value", [
    ("c", .5), ("half_life", 4), ("cadence", 2), ("rank", 4),
    ("score_samples", 128), ("retention", {"omega": 39}), ("episode_arrivals", 16),
])
def test_same_policy_replica_is_separated_by_scientific_protocol(tmp_path, setting, value):
    root = make_root(tmp_path)
    detail = {"policy": {"c": 1., "half_life": 8., "cadence": 1, "rank": 8, "score_samples": 64},
              "retention": {"omega": 19}, "episode_arrivals": 32}
    first = seal_unit(root, phase="phase3", policy="asg_lr_lr", detail=detail)
    first.rename(first.with_name("first-candidate"))
    if setting in detail:
        detail[setting] = value
    else:
        detail["policy"][setting] = value
    seal_unit(root, phase="phase3", policy="asg_lr_lr", detail=detail)
    report = analysis.load_report(root)
    groups = analysis.trajectory_groups(report)
    assert len(groups) == 2
    assert len({analysis.unit_metadata(group[0])["protocol_variant"] for group in groups}) == 2
    assert all(analysis.pointwise_summary(group, "current_nll")[0]["n"] == 1 for group in groups)


def test_pr_replica_interpolation_precedes_averaging_and_ties_are_explicit():
    def unit(index, curve):
        return {"config": {"kind": "trajectory", "index": index, "policy": "asg"}, "summary": {},
                "pr_curves": [{"step": 1, "threshold": threshold, "recall": recall,
                               "precision": precision, "precision_defined": True}
                              for threshold, recall, precision in curve]}
    units = [unit(0, [(.9, .2, 1.), (.5, .8, .6), (.2, .8, .4)]),
             unit(1, [(.8, .4, .8), (.1, .6, .2)])]
    rows = analysis.pr_curve_summary(units, points=11)
    # Per-replica precision at recall .5 is .7 and .5; no threshold pooling.
    assert rows[5]["recall"] == .5
    assert rows[5]["mean"] == pytest.approx(.6)
    assert rows[5]["n"] == 2
    assert rows[8]["mean"] == .4 and rows[8]["n"] == 1


def test_canonical_fixed_controls_are_reused_without_pooling_active_candidates(tmp_path):
    root = make_root(tmp_path)
    for name, gate, c, cadence, h in [("depth6_lr", None, 1., 1, 8.),
                                     ("asg_lr_lr", "lr", 2., 2, 8.),
                                     ("asg_lr_lr", "lr", 4., 2, 8.),
                                     ("depth6_diag", None, 1., 1, 4.)]:
        unit = seal_unit(root, phase="phase3", policy=name, detail={"policy": {
            "gate": gate, "c": c, "cadence": cadence, "half_life": h}})
        unit.rename(unit.with_name(f"{name}-{c}-{h}"))
    groups = analysis.trajectory_groups(analysis.load_report(root))
    candidate_groups = [group for group in groups if any(unit["config"]["policy"] == "asg_lr_lr" for unit in group)]
    assert len(candidate_groups) == 2
    for group in candidate_groups:
        assert {unit["config"]["policy"] for unit in group} == {"asg_lr_lr", "depth6_lr"}
        assert len(group) == 2
        assert len(analysis.pointwise_summary(group, "current_nll")) == 8


def test_integrity_cache_reuses_only_unchanged_payloads_and_force_fresh_rehashes(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    unit = seal_unit(root)
    original_open = Path.open
    reads = []
    def track_open(path, mode="r", *args, **kwargs):
        if mode == "rb":
            reads.append(path)
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", track_open)
    analysis.load_report(root)
    initial_reads = len(reads)
    assert initial_reads >= 3
    analysis.load_report(root)
    assert len(reads) == initial_reads
    analysis.load_report(root, force_fresh=True)
    assert len(reads) == 2 * initial_reads
    path = unit / "summary.json"
    old_stat = path.stat()
    path.write_text(path.read_text().replace('"nll_auc": 0.8', '"nll_auc": 0.7'))
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    with pytest.raises(analysis.ArtifactError, match="Corrupt artifact"):
        analysis.load_report(root)
    assert len(reads) > 2 * initial_reads


def test_cached_payload_does_not_cache_manifest_or_expected_digest(tmp_path):
    root = make_root(tmp_path)
    unit = seal_unit(root)
    analysis.load_report(root)
    integrity = json.loads((unit / "integrity.json").read_text())
    integrity["summary.json"] = "0" * 64
    write_json(unit / "integrity.json", integrity)
    with pytest.raises(analysis.ArtifactError, match="Corrupt artifact"):
        analysis.load_report(root)
    reseal(unit)
    manifest = json.loads((unit / "manifest.json").read_text())
    manifest["metric_schema_version"] = 9
    write_json(unit / "manifest.json", manifest)
    with pytest.raises(analysis.ArtifactError, match="metric_schema_version"):
        analysis.load_report(root)


def test_frozen_calibration_pointer_is_checked_against_sealed_decision(tmp_path):
    root = make_root(tmp_path)
    selected = {"c": 2., "half_life": 8., "cadence": 1}
    frozen = {"schema_version": 1, "selected": selected}
    unit = seal_unit(root, phase="phase3", kind="decision", policy="phase3_calibration",
                     summary={"decision": "phase3_calibration", **frozen})
    write_json(unit / "decision.json", frozen)
    reseal(unit)
    write_json(root / "decisions/phase3_calibration.json", frozen)
    assert analysis.load_report(root)["decisions"]["phase3_calibration"]["selected"] == selected
    write_json(root / "decisions/phase3_calibration.json", {"schema_version": 1, "selected": {**selected, "c": 4.}})
    with pytest.raises(analysis.ArtifactError, match="disagrees with sealed decision"):
        analysis.load_report(root)
