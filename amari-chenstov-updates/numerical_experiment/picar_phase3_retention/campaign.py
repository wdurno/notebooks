"""Conditional retention repair inside the original shared Phase 3 allowance."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.report import read_completed_unit
from numerical_experiment.picar.runtime import StopController, check_storage, sha256_file
from numerical_experiment.picar.storage import RunCollisionError, RunStore, _safe_child, _verify_file
from numerical_experiment.picar_phase2.campaign import PROJECT, audit_initial, load_completed_state, read_json
from numerical_experiment.picar_phase3.analysis import candidate, forecast, select_control, selection
from numerical_experiment.picar_phase3.campaign import learning_metadata
from numerical_experiment.picar_phase3.config import (
    DEVELOPMENT_SEEDS, MAX_HOURS, MODEL_SEED, NORMALIZATION, PRIOR_STUDY_SECONDS,
    VERSION as SOURCE_VERSION, Recipe, configuration, likelihood_key, panels,
    recipe_dict, simulation, sources as old_sources,
)


VERSION = "plan19-phase3-retention-v1"
CONTROL_NAMES = tuple(f"control_ridge_{ridge:g}" for ridge in (.1, .03, .3))
CLOSEOUT_SECONDS = 300.
STOP_MARGIN_SECONDS = 90.
SCREEN_POLL_SECONDS = 10.


def scientific_sources():
    paths = [*Path(__file__).parent.glob("*.py"), PROJECT / "test/unit/test_picar_phase3_retention.py"]
    return {**old_sources(), **{str(p.relative_to(PROJECT)): sha256_file(p) for p in sorted(paths)}}


def _recipe(summary):
    return Recipe(**{k: v for k, v in summary["settings"].items() if k != "seed"})


def _source_config(source):
    parent = read_json(source / "orchestration/config.json")
    if (parent["version"] != SOURCE_VERSION or parent["phase"] != 3
            or parent["source_sha256"] != old_sources()
            or parent["prior_study_seconds"] != PRIOR_STUDY_SECONDS
            or not 0 < parent["max_hours"] <= MAX_HOURS):
        raise ValueError("source campaign version, sources or shared allowance differs")
    return parent


def _screen_ready(source):
    # A cheap discovery gate, not evidence validation. Configs and completion
    # markers are immutable once published; all seals are checked below later.
    names = set()
    for marker in (source / "units").glob("pilot-*/COMPLETED"):
        settings = read_json(marker.parent / "config.json").get("settings", {})
        if settings.get("name") in CONTROL_NAMES and settings.get("seed") == DEVELOPMENT_SEEDS[0]:
            names.add(settings["name"])
    return names == set(CONTROL_NAMES)


def screen_controls(source):
    """Read sealed outcomes only; never stop a source with an incomplete screen."""
    source = Path(source)
    parent = _source_config(source)
    found = {}
    for marker in sorted((source / "units").glob("pilot-*/COMPLETED")):
        unit = read_completed_unit(marker.parent)
        value, stored = unit["summary"], unit["stored_config"]
        name = value.get("settings", {}).get("name")
        if name not in CONTROL_NAMES:
            continue
        seed = value.get("seed")
        if seed not in DEVELOPMENT_SEEDS or value["settings"].get("seed") != seed:
            raise ValueError("source controls must use the fixed development seeds")
        if seed != DEVELOPMENT_SEEDS[0]:
            continue
        if name in found or any(stored.get(k) != v for k, v in parent.items()):
            raise ValueError("duplicate control or unknown source configuration/version")
        recipe = _recipe(value)
        recipe.validate()
        if (recipe.fd_strength != 0 or name != f"control_ridge_{recipe.ridge:g}"
                or stored.get("settings") != value["settings"]
                or Path(value["unit_path"]).resolve() != marker.parent.resolve()):
            raise ValueError("source control recipe/provenance differs")
        candidate(value)
        found[name] = value
    controls = [found[n] for n in CONTROL_NAMES if n in found]
    common = [recipe_dict(replace(_recipe(p), name="screen", ridge=.1)) for p in controls]
    if common and any(c != common[0] for c in common):
        raise ValueError("source ridge screen changes other common settings")
    return parent, controls


def takeover_required(controls):
    if len(controls) != len(CONTROL_NAMES):
        raise RuntimeError("all three completed ridge controls are required before takeover")
    _, rows = select_control(controls)
    feasible = any(r["new_gain_endpoint"] >= .1 - 1e-12
                   and r["old_gap_endpoint"] <= .1 + 1e-12 for r in rows)
    return not feasible


def _partial_progress(path):
    """Inspect the committed JSON cursor without opening/pruning an old run."""
    if not (path / "CURRENT.json").exists():
        return {}
    pointer = read_json(path / "CURRENT.json")
    directory = _safe_child(path, f"transactions/{pointer['commit_id']}")
    _verify_file(directory / "manifest.json", pointer["manifest"])
    manifest = read_json(directory / "manifest.json")
    _verify_file(directory / "progress.json", manifest["files"]["progress.json"])
    return read_json(directory / "progress.json")


def _preserved_attempts(source):
    preserved, rounds = [], set()
    for path in sorted((source / "units").glob("pilot-*")):
        settings = read_json(path / "config.json")["settings"]
        if settings["name"] in CONTROL_NAMES and settings["seed"] == DEVELOPMENT_SEEDS[0]:
            continue
        progress = _partial_progress(path)
        count = progress.get("counts", {}).get("step", 0)
        if type(count) is not int or count < 0:
            raise ValueError("invalid preserved trajectory acquisition count")
        name = settings["name"]
        if count > 0 and (name == "common_plasticity_repair"
                          or name.startswith("fd_ratio_") and name != "fd_ratio_01"):
            rounds.add(name)
        preserved.append({"path": str(path), "settings": settings, "acquired_count": count,
            "completed": (path / "COMPLETED").exists(), "used_for_selection": False,
            "current_sha256": sha256_file(path / "CURRENT.json") if (path / "CURRENT.json").exists() else None})
    return preserved, sorted(rounds)


def seal_source(source, output, *, wait=False, stopped=lambda: False):
    source, output = Path(source).resolve(), Path(output).resolve()
    _source_config(source)
    while True:
        if stopped():
            raise InterruptedError("retention supervisor stopped while waiting; source left intact")
        if _screen_ready(source):
            break
        if not wait or (source / "orchestration/COMPLETED").exists():
            raise RuntimeError("all three completed ridge controls are required; use --wait")
        time.sleep(SCREEN_POLL_SECONDS)
    config, controls = screen_controls(source)
    if not takeover_required(controls):
        return {"status": "original_campaign_retained", "takeover": False,
                "candidate_rows": [candidate(p) for p in controls]}
    path = source / "orchestration"
    # Acquire the authoritative writer lock through RunStore, avoiding a
    # check-unlock-reopen race. Waiting overlaps the already charged old run.
    if not (path / "COMPLETED").exists():
        (source / "STOP").touch(exist_ok=True)
    while True:
        try:
            store = RunStore(path, config, resume=True, checkpoint_policy="journal-v1")
            break
        except RunCollisionError as exc:
            if "another writer" not in str(exc):
                raise
            if stopped():
                raise InterruptedError("source STOP requested; resume supervisor to finish handover")
            time.sleep(.5)
    with store:
        if store.completed:
            result = read_completed_unit(path)["summary"]
            if result.get("status") != "superseded_after_ridge_screen":
                return {"status": "original_campaign_completed", "takeover": False}
            if result["superseded_by"] != str(output):
                raise ValueError("source was superseded by a different supervisor output")
            return result
        saved = store.load_checkpoint(trusted=True)
        charged = saved["charged_seconds"]
        if not math.isfinite(charged) or charged < 0:
            raise ValueError("source durable charged_seconds is invalid")
        preserved, spent = _preserved_attempts(source)
        result = {"schema_version": 1, "phase": 3, "kind": "phase3_campaign", "smoke_only": True,
            "status": "superseded_after_ridge_screen", "takeover": True,
            "superseded_by": str(output), "total_active_seconds": charged,
            "phase3_active_seconds": charged, "cumulative_study_seconds": PRIOR_STUDY_SECONDS + charged,
            "active_seconds": 0., "control_paths": [p["unit_path"] for p in controls],
            "preserved_trajectories": preserved, "spent_repair_rounds": spent,
            "next_phase_authorized": False}
        store.complete(result)
        return result


def _validated_panels(recipe, stored, marker):
    """Artifact-only checks; importing the learning runtime would import Torch."""
    specs = panels(recipe)
    by_name = {p["difficulty"]: p for p in stored}
    if len(by_name) != len(stored) or set(by_name) != {p["difficulty"] for p in specs}:
        raise ValueError("initial panel difficulties differ")
    for spec in specs:
        panel = by_name[spec["difficulty"]]
        seeds, rows = list(spec["seeds"]), panel["rows"]
        if (panel.get("checkpoint") != "initial" or panel.get("acquired_count") != 0
                or panel.get("initial_asset_id") != marker or panel.get("seeds") != seeds
                or canonical_json(panel.get("config")) != canonical_json(asdict(spec["config"]))
                or [r["seed"] for r in rows] != seeds):
            raise ValueError("initial panel configuration/asset/seeds differ")
        if any(r.get("complete") is not True or r.get("truncated") is not False
               or r.get("evaluation") is not True or r.get("learning") is not False
               or r.get("explore") is not False or r.get("reward_scoring") is not False
               or r.get("actor") != "learned_student" or r.get("acquired_count") != 0 for r in rows):
            raise ValueError("initial panel rows must be complete frozen initial evaluations")
    return copy.deepcopy([by_name[p["difficulty"]] for p in specs])


def source_initial(source, controls):
    first = controls[0]
    recipe = _recipe(first)
    asset_ids = {p.get("initial_asset_id") for c in controls for p in c["evaluations"]
                 if p["checkpoint"] == "initial"}
    if len(asset_ids) != 1 or None in asset_ids:
        raise ValueError("source controls do not identify one common initial asset")
    asset_id = next(iter(asset_ids))
    matches = [p.parent for p in (source / "units").glob("bootstrap-*/COMPLETED")
               if sha256_file(p) == asset_id]
    if len(matches) != 1:
        raise ValueError("source bootstrap marker is missing or ambiguous")
    unit = read_completed_unit(matches[0])
    if unit["stored_config"]["settings"]["likelihood_key"] != likelihood_key(recipe):
        raise ValueError("source initial likelihood/normalization differs")
    initial_evaluations = [p for p in first["evaluations"] if p["checkpoint"] == "initial"]
    _validated_panels(recipe, initial_evaluations, asset_id)
    for c in controls[1:]:
        if likelihood_key(_recipe(c)) != likelihood_key(recipe):
            raise ValueError("cannot mix source initial likelihoods")
        _validated_panels(recipe, [p for p in c["evaluations"] if p["checkpoint"] == "initial"], asset_id)
    return matches[0], asset_id, initial_evaluations, unit["summary"]


def choose_control(controls, gentle):
    old, _ = select_control(controls)
    original = next(p for p in controls if p["settings"]["name"] == old["name"])
    if gentle is None:
        return _recipe(original)
    new = candidate(gentle)
    feasible = new["new_gain_endpoint"] >= .1 - 1e-12 and new["old_gap_endpoint"] <= .1 + 1e-12
    improves = (new["old_gap_endpoint"] < old["old_gap_endpoint"] - 1e-12
                and new["old_gap_average"] <= old["old_gap_average"] + 1e-12)
    return _recipe(gentle if feasible or improves else original)


class SharedBudget:
    def __init__(self, old_seconds, new_seconds=0., *, limit=MAX_HOURS * 3600, clock=None):
        if any(not math.isfinite(x) or x < 0 for x in (old_seconds, new_seconds, limit)):
            raise ValueError("invalid shared budget")
        self.old, self.base, self.limit = old_seconds, new_seconds, limit
        self.clock = clock or time.monotonic
        self.started = self.clock()

    def new(self):
        return self.base + self.clock() - self.started

    def total(self):
        return self.old + self.new()

    def admit(self, estimate, *, partial=False):
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError("admission estimate must be finite and nonnegative")
        remaining = min(self.limit, 72000 - PRIOR_STUDY_SECONDS) - self.total() - CLOSEOUT_SECONDS
        return remaining > STOP_MARGIN_SECONDS and (partial or remaining >= 1.2 * estimate + 300.)


def _check_model(actual, expected):
    fields = ("kind", "declared_revision", "checkpoint_manifest", "processor_pixels", "max_tokens",
        "quantization", "software", "seeds", "selected_layers", "adapter_targets", "factor_scales",
        "initialization", "model_schema", "compute_policy", "trainable_inventory")
    if (not isinstance(expected, dict) or not expected.get("checkpoint_manifest", {}).get("sha256")
            or any(canonical_json(actual.get(k)) != canonical_json(expected.get(k)) for k in fields)):
        raise ValueError("frozen model bits or execution design differ from the source bootstrap")


class _Execution:
    def __init__(self, model_path, offline, *, reference_model=None):
        self.model_path, self.offline = model_path, offline
        self.reference_model = reference_model
        self.model, self.model_key, self.states = None, None, {}

    def model_for(self, recipe):
        import torch
        key = likelihood_key(recipe)
        if self.model is not None and self.model_key != key:
            import gc
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
        if self.model is None:
            from numerical_experiment.picar.models import load_qwen_model
            from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
            torch.set_num_threads(2)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no CPU experiment substitution")
            kwargs = {} if self.model_path is None else {"model_path": self.model_path}
            raw = load_qwen_model(**kwargs, local_files_only=self.offline, seed=MODEL_SEED,
                top_layers=1, ltm_rank=4, stm_rank=1, hidden_size=64, context_length=4,
                return_bins=161, min_pixels=56*56, max_pixels=19200, max_tokens=2048).eval()
            _check_model(raw.encoder.provenance, self.reference_model)
            raw.encoder.enable_prefix_cache = True
            self.model = NormalizedStudent(raw, mode=NORMALIZATION, eps=1e-6).eval()
            self.model_key = key
        return self.model

    def initial_for(self, recipe, asset):
        marker = sha256_file(asset / "COMPLETED")
        if marker not in self.states:
            self.states[marker] = load_completed_state(asset)
        initial = self.states[marker]
        audit_initial(initial, configuration(recipe))
        provenance = initial["identity"]["phase3"]["model_provenance"]
        if provenance["mode"] != NORMALIZATION or provenance["eps"] != 1e-6:
            raise ValueError("bootstrap normalization differs")
        return initial, marker

    def bootstrap(self, recipe, store, progress, teacher_source):
        from numerical_experiment.picar_phase2_rescue.data import restore_demonstrations
        from numerical_experiment.picar_phase3.student import DevelopmentRun
        run = DevelopmentRun(self.model_for(recipe), configuration(recipe, bootstrap=True),
            simulation(recipe, bootstrap=True), store, progress, mode="bootstrap",
            seed=configuration(recipe, bootstrap=True).seed, max_transitions=600)
        if run.s["stage"] == "collect":
            data = restore_demonstrations(teacher_source)
            run.s.update(eligible=copy.deepcopy(data["records"]), retired=[], ready=[],
                acquired_count=data["acquired_count"], episodes=copy.deepcopy(data["episodes"]),
                initial_data_provenance=data["provenance"], stage="distill")
            run.save()
        return {**run.run(), "data_provenance": run.s["initial_data_provenance"],
            "normalization": NORMALIZATION, "old_fisher_reused": False, "new_teacher_transitions": 0,
            "execution": learning_metadata(run.learner), "likelihood_key": likelihood_key(recipe)}

    def pilot(self, recipe, seed, store, progress, asset, initial_evaluations):
        initial, marker = self.initial_for(recipe, asset)
        from numerical_experiment.picar_phase3.student import DevelopmentRun
        run = DevelopmentRun(self.model_for(recipe), configuration(recipe), simulation(recipe), store, progress,
            mode="pilot", seed=seed, max_transitions=recipe.transitions, initial=initial,
            evaluations=panels(recipe), initial_evaluations=initial_evaluations, initial_asset_id=marker)
        return {**run.run(), "execution": learning_metadata(run.learner),
                "initial_asset_marker_sha256": marker,
                "initialization": {"source_weights": str(asset), "marker_sha256": marker,
                    "bootstrap_execution": read_json(asset / "summary.json").get("execution"),
                    "kl_repeated_in_pilot": False, "old_phase2_fisher_reused": False}}

    def probe(self, recipe, asset):
        initial, marker = self.initial_for(recipe, asset)
        from numerical_experiment.picar_phase3.calibration import fd_probe
        return {**fd_probe(self.model_for(recipe), configuration(recipe), initial),
                "initial_asset_marker_sha256": marker}


def _software(store):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q",
        "test/unit/test_picar_phase3_retention.py", "test/unit/test_picar_phase3_student.py",
        "test/unit/test_picar_phase3_analysis.py", "test/unit/test_picar_phase3_probes.py"],
        cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    (store.root / "pytest.txt").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"retention software acceptance failed: {store.root / 'pytest.txt'}")
    return {"passed": True, "test_result": result.stdout.strip().splitlines()[-1]}


def run(output, source_campaign, *, model_path=None, offline=False, resume=False, wait=False, stage_limit="all"):
    if stage_limit not in {"software", "all"}:
        raise ValueError("stage must be software or all")
    root, source = Path(output).resolve(), Path(source_campaign).resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("supervisor and source campaign must have disjoint artifact directories")
    if (root / "orchestration").exists() and not resume:
        raise RunCollisionError("supervisor already exists; use --resume")
    original_config = read_json(source / "orchestration/config.json")
    model_source = str(Path(model_path).resolve()) if model_path else configuration(Recipe("identity")).model.model_id
    if model_path is not None and not Path(model_source).is_dir():
        raise ValueError("model source directory is unavailable")
    stop = StopController(root)
    stop.install()
    try:
        takeover = seal_source(source, root, wait=wait, stopped=stop.check)
        if not takeover.get("takeover"):
            return takeover
        started = time.monotonic()
        source_config, controls = screen_controls(source)
        asset, asset_id, initial_evaluations, bootstrap = source_initial(source, controls)
        chosen, _ = select_control(controls)
        original = next(p for p in controls if p["settings"]["name"] == chosen["name"])
        gentle_recipe = replace(_recipe(original), name="control_retention_repair", learning_rate=.001,
                                optimizer_steps=1, pi=.01, exploration_std=4., action_variance=16.)
        hashes = scientific_sources()
        identity = {"schema_version": 1, "version": VERSION, "phase": 3, "source_sha256": hashes,
            "source_campaign": str(source), "source_campaign_marker_sha256": sha256_file(source / "orchestration/COMPLETED"),
            "source_config_sha256": sha256_file(source / "orchestration/config.json"),
            "source_initial_asset": str(asset), "source_initial_asset_marker_sha256": asset_id,
            "source_initial_execution": bootstrap.get("execution"), "model_source": model_source,
            "source_model_source": original_config["model_source"],
            "teacher_source": source_config["teacher_source"],
            "teacher_completion_sha256": source_config["teacher_completion_sha256"],
            "old_phase3_seconds": takeover["phase3_active_seconds"], "prior_study_seconds": PRIOR_STUDY_SECONDS,
            "max_hours": source_config["max_hours"], "normalization": NORMALIZATION,
            "config": configuration(gentle_recipe).to_dict(), "gentle_recipe": recipe_dict(gentle_recipe),
            "control_choice": "feasible_or_strictly_better_endpoint_retention_with_nonworse_average_else_old",
            "maximum_repair_rounds": 2, "new_prevalidation_reserve_seconds": 0.,
            "closeout_reserve_seconds": CLOSEOUT_SECONDS, "next_phase_authorized": False}
        root.mkdir(parents=True, exist_ok=True)
        with RunStore(root / "orchestration", identity, resume=resume, checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return read_completed_unit(parent.root)["summary"]
            saved = parent.load_checkpoint(trusted=True)
            state = saved or {"new_charged_seconds": 0., "budget_skips": [],
                "repair_rounds": [{"round": i+1, "name": name, "origin": "source_campaign", "preserved": True}
                                  for i, name in enumerate(takeover["spent_repair_rounds"])]}
            budget = SharedBudget(identity["old_phase3_seconds"], state["new_charged_seconds"],
                                  limit=identity["max_hours"] * 3600)
            if saved is None:
                budget.started = started
            active, last_save, last_report = None, 0., 0.
            stage = "retention repair acceptance"

            def refresh():
                from numerical_experiment.picar.report import build_notebook
                build_notebook(PROJECT / "cache/numerical_experiment/picar")

            def progress(label=None, *, force=False, check=True):
                nonlocal stage, last_save, last_report
                stage = label or stage
                now = time.monotonic()
                if force or now-last_save >= 15 or stop.check():
                    new_seconds = budget.new()
                    total_seconds = budget.old + new_seconds
                    state["new_charged_seconds"] = new_seconds
                    child = _partial_progress(active) if active and (active / "CURRENT.json").exists() else {}
                    parent.checkpoint(copy.deepcopy(state), progress={"phase": 3, "status": stage,
                        "phase3_active_seconds": total_seconds, "total_active_seconds": new_seconds,
                        "cumulative_study_seconds": PRIOR_STUDY_SECONDS + total_seconds,
                        "remaining_budget_seconds": max(0., budget.limit-total_seconds-CLOSEOUT_SECONDS),
                        "completed_trajectories": len(list((root / "units").glob("pilot-*/COMPLETED"))),
                        "historical_reference_trajectories": len(controls),
                        "counts": {"step": len(list((root / "units").glob("*/COMPLETED"))), "target": 13},
                        "active_unit": str(active) if active else None, "active_unit_progress": child})
                    last_save = now
                if force or now-last_report >= 900:
                    refresh()
                    last_report = now
                if check and stop.check():
                    raise InterruptedError("retention repair checkpointed; remove STOP before --resume")
                if check and not budget.admit(0., partial=True):
                    raise TimeoutError("shared Phase 3 ceiling reached; partial artifacts preserved")

            def unit_path(name, settings):
                config = {**identity, "operation": name, "settings": settings}
                if name in {"pilot", "probe"}:
                    values = settings if name == "pilot" else settings["recipe"]
                    recipe = Recipe(**{k: v for k, v in values.items() if k != "seed"})
                    config.update(config=configuration(recipe).to_dict(),
                                  initial_asset_likelihood_key=likelihood_key(recipe))
                elif name == "bootstrap":
                    config["config"] = configuration(gentle_recipe, bootstrap=True).to_dict()
                suffix = hashlib.sha256(canonical_json(config).encode()).hexdigest()[:12]
                return root / "units" / f"{name}-{suffix}", config

            def unit(name, settings, callback):
                nonlocal active
                path, config = unit_path(name, settings)
                active = path
                # Completed work can be read even after admission closes.
                if (path / "COMPLETED").exists():
                    value = read_completed_unit(path)
                    expected = {**config, "checkpoint_policy": parent.config["checkpoint_policy"]}
                    if canonical_json(value["stored_config"]) != canonical_json(expected):
                        raise ValueError("completed supervisor unit identity differs")
                    active = None
                    return value["summary"]
                progress(name, force=True)
                with RunStore(path, config, resume=resume, checkpoint_policy="journal-v1") as store:
                    tick = time.monotonic()
                    result = callback(store)
                    if scientific_sources() != hashes:
                        raise RuntimeError("scientific source changed; preserve partial and version correction")
                    result = {**result, "schema_version": 1, "phase": 3, "kind": f"phase3_{name}",
                        "supervisor_version": VERSION, "smoke_only": True, "development_only": True,
                        "settings": settings, "unit_path": str(path), "active_seconds": time.monotonic()-tick,
                        "source_campaign_marker_sha256": identity["source_campaign_marker_sha256"],
                        "source_bootstrap_marker_sha256": asset_id}
                    if store._current is None:
                        store.checkpoint({"stage": "done"}, progress={"phase": 3, "status": name})
                    store.complete(result)
                active = None
                progress(name + " complete", force=True)
                return result

            def archive(store):
                path = store.root / "sources.tar.gz"
                with tarfile.open(path, "w:gz") as tar:
                    for name in hashes:
                        tar.add(PROJECT / name, arcname=name)
                return {"source_sha256": hashes, "archive_sha256": sha256_file(path)}

            def skip(settings, reason, estimate):
                row = {"settings": settings, "reason": reason, "estimated_seconds": estimate,
                       "remaining_seconds": max(0., budget.limit-budget.total()-CLOSEOUT_SECONDS)}
                state["budget_skips"] = [r for r in state["budget_skips"] if r["settings"] != settings] + [row]
                progress("budget admission skipped", force=True)

            execution = _Execution(model_path, offline,
                reference_model=(bootstrap.get("execution") or {}).get("model_provenance"))
            initials = {likelihood_key(_recipe(original)): {"asset": asset, "panels": initial_evaluations}}
            pilots, recipes = list(controls), {_recipe(p).name: _recipe(p) for p in controls}
            measured_old = max(p["timings"]["active_seconds"] for p in controls)
            initialization_estimate = max(840., bootstrap.get("timings", {}).get("active_seconds", 0.))

            def initial_for(recipe):
                key = likelihood_key(recipe)
                if key not in initials:
                    settings = {"name": "common_initialization", "likelihood_key": key}
                    def initialize(store):
                        teacher_source = Path(identity["teacher_source"])
                        if sha256_file(teacher_source / "COMPLETED") != identity["teacher_completion_sha256"]:
                            raise ValueError("original demonstration completion marker differs")
                        return execution.bootstrap(recipe, store, progress, teacher_source)
                    value = unit("bootstrap", settings, initialize)
                    initials[key] = {"asset": Path(value["unit_path"]), "panels": None}
                return initials[key]

            def pilot(recipe, seed, *, repair=None, fd_multiplier=1.):
                settings = {**recipe_dict(recipe), "seed": seed}
                path, _ = unit_path("pilot", settings)
                completed, partial = (path / "COMPLETED").exists(), (path / "config.json").exists()
                observed = [p["timings"]["active_seconds"] for p in pilots
                            if p["settings"]["name"] == recipe.name]
                pilot_estimate = max(observed, default=measured_old * fd_multiplier)
                estimate, initializing = pilot_estimate, False
                if likelihood_key(recipe) not in initials:
                    bootstrap_path, _ = unit_path("bootstrap", {"name": "common_initialization",
                                                               "likelihood_key": likelihood_key(recipe)})
                    if not (bootstrap_path / "COMPLETED").exists():
                        estimate += initialization_estimate
                        initializing = (bootstrap_path / "config.json").exists()
                if not completed and not budget.admit(estimate, partial=partial or initializing):
                    skip(settings, "shared Phase 3 admission allowance", estimate)
                    return None
                if repair and not any(r["name"] == recipe.name for r in state["repair_rounds"]):
                    if len(state["repair_rounds"]) >= 2:
                        skip(settings, "two shared repair rounds already spent", estimate)
                        return None
                    state["repair_rounds"].append({"round": len(state["repair_rounds"])+1,
                        "name": recipe.name, "origin": VERSION, "purpose": repair, "recipe": recipe_dict(recipe)})
                    progress("repair round admitted", force=True)
                initialized = initial_for(recipe)
                if not completed and not partial and not budget.admit(pilot_estimate):
                    skip(settings, "post-initialization trajectory admission allowance", pilot_estimate)
                    return None
                value = unit("pilot", settings, lambda store: execution.pilot(recipe, seed, store, progress,
                    initialized["asset"], initialized["panels"]))
                if initialized["panels"] is None:
                    marker = sha256_file(initialized["asset"] / "COMPLETED")
                    initial_rows = [p for p in value["evaluations"] if p["checkpoint"] == "initial"]
                    initialized["panels"] = _validated_panels(recipe, initial_rows, marker)
                pilots.append(value)
                recipes[recipe.name] = recipe
                return value

            try:
                unit("provenance", {}, archive)
                software_path, _ = unit_path("software", {})
                if not (software_path / "COMPLETED").exists() and not budget.admit(600.):
                    raise TimeoutError("shared allowance cannot admit bounded CPU software acceptance")
                unit("software", {}, _software)
                references = unit("reference_controls", {}, lambda store: {
                    "candidate_rows": [candidate(p) for p in controls],
                    "source_paths": [p["unit_path"] for p in controls], "source_campaign": str(source),
                    "new_trajectories": False, "already_charged": True})
                if stage_limit == "software":
                    progress("software accepted; retention trajectories pending", force=True)
                    return {"status": "software_complete", "phase3_active_seconds": budget.total(),
                            "total_active_seconds": budget.new()}
                check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)
                gentle = pilot(gentle_recipe, DEVELOPMENT_SEEDS[0], repair="Shared gentler retention-directed control")
                control = choose_control(controls, gentle)
                pilot(control, DEVELOPMENT_SEEDS[1])
                probe_settings = {"name": "fd_resolution_and_cost", "recipe": recipe_dict(control)}
                probe_path, _ = unit_path("probe", probe_settings)
                numerical = None
                if (probe_path / "config.json").exists() or budget.admit(600.):
                    initialized = initial_for(control)
                    numerical = unit("probe", probe_settings, lambda store: execution.probe(control, initialized["asset"]))
                else:
                    skip(probe_settings, "FD probe allowance unavailable", 600.)
                if numerical and numerical["fd_convergence_pass"]:
                    rows = numerical["fd_strength_rows"]
                    middle = next(r for r in rows if r["target_gradient_ratio"] == .1)
                    fd = replace(control, name="fd_retention_ratio_01", fd_strength=middle["fd_strength"],
                                 fd_step=numerical["selected_fd_step"])
                    costs = numerical["fit_cost_rows"]
                    times = {r["fd_strength"]: r["median_update_seconds"] for r in costs}
                    multiplier = max(1., times[fd.fd_strength] / times[0.])
                    first = pilot(fd, DEVELOPMENT_SEEDS[0], fd_multiplier=multiplier)
                    if first is not None:
                        pilot(fd, DEVELOPMENT_SEEDS[1], fd_multiplier=multiplier)
                        comparison = selection(pilots, control_name=control.name)
                        relevant = [r for r in comparison["comparisons"] if r["name"] == fd.name]
                        if relevant and not all(r["retention_matched"] for r in relevant):
                            gap = sum(r["retention_endpoint_difference"] for r in relevant) / len(relevant)
                            target = 1. if gap > 0 else .01
                            adjacent = next(r for r in rows if r["target_gradient_ratio"] == target)
                            alternate = replace(fd, name=f"fd_retention_ratio_{target:g}", fd_strength=adjacent["fd_strength"])
                            adjusted = pilot(alternate, DEVELOPMENT_SEEDS[0], fd_multiplier=multiplier,
                                             repair="One adjacent retention-qualified FD strength adjustment")
                            if adjusted is not None:
                                pilot(alternate, DEVELOPMENT_SEEDS[1], fd_multiplier=multiplier)
                decision = selection(pilots, control_name=control.name)
                decision.update(selected_control_recipe=recipe_dict(control),
                    selected_fd_recipe=recipe_dict(recipes[decision["selected_fd"]]) if decision["selected_fd"] else None,
                    repair_rounds=state["repair_rounds"], budget_skips=state["budget_skips"],
                    reference_controls=references["unit_path"], historical_reference_paths=[p["unit_path"] for p in controls],
                    fd_convergence_pass=numerical["fd_convergence_pass"] if numerical else None,
                    confirmation_proposal={"authorized": False, "requires_fresh_review": True})
                selected = unit("selection", {}, lambda store: decision)
                projected = forecast(pilots, decision, phase3_seconds=budget.total()+CLOSEOUT_SECONDS,
                                     prior_study_seconds=PRIOR_STUDY_SECONDS)
                resource = unit("forecast", {}, lambda store: {**projected, "closeout_reserve_seconds": CLOSEOUT_SECONDS})
                progress("retention-directed development complete; check-in only", force=True)
                processing_seconds = budget.new()
                new_charge = processing_seconds + CLOSEOUT_SECONDS
                total = budget.old + new_charge
                if total > min(budget.limit, 72000 - PRIOR_STUDY_SECONDS):
                    raise TimeoutError("closeout would exceed the shared allowance")
                result = {"schema_version": 1, "phase": 3, "kind": "phase3_campaign", "smoke_only": True,
                    "supervisor_version": VERSION, "status": "ready_for_checkin", "next_phase_authorized": False,
                    "total_active_seconds": new_charge, "phase3_active_seconds": total,
                    "cumulative_study_seconds": PRIOR_STUDY_SECONDS+total, "active_seconds": 0.,
                    "old_phase3_seconds": budget.old, "closeout_reserve_seconds": CLOSEOUT_SECONDS,
                    "completed_trajectories": len(list((root / "units").glob("pilot-*/COMPLETED"))),
                    "historical_reference_trajectories": len(controls),
                    "units": len(list((root / "units").glob("*/COMPLETED"))),
                    "pilot_paths": [p["unit_path"] for p in pilots
                                    if p["unit_path"] not in {x["unit_path"] for x in controls}],
                    "reference_controls": references["unit_path"], "selection": selected, "forecast": resource}
                # Recovery before COMPLETED must not add the closeout reserve
                # twice. Persist processing and the one reserve separately.
                state["new_charged_seconds"] = processing_seconds
                state["closeout_reserve_seconds"] = CLOSEOUT_SECONDS
                parent.checkpoint(state, progress={"phase": 3, "status": result["status"],
                    "phase3_active_seconds": total, "total_active_seconds": result["total_active_seconds"],
                    "completed_trajectories": result["completed_trajectories"],
                    "historical_reference_trajectories": len(controls),
                    "counts": {"step": result["units"], "target": result["units"]},
                    "cumulative_study_seconds": result["cumulative_study_seconds"]})
                parent.complete(result)
                refresh()
                return result
            except BaseException:
                progress("checkpointed interruption; explicit resume required", force=True, check=False)
                raise
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cache/numerical_experiment/picar/phase3/development-retention-v1"))
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--stage", choices=("software", "all"), default="all")
    args = parser.parse_args()
    result = run(args.output, args.source_campaign, model_path=args.model_path,
                 offline=args.offline, resume=args.resume, wait=args.wait, stage_limit=args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
