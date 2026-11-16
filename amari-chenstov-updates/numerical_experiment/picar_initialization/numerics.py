"""Bounded stationary LTM fitting and equal-weight, fixed-anchor model scores.

The caller owns immutable full causal histories and the whole-episode split.
Checkpoints contain references/identities, not another copy of those histories.
Scores are gradients of log likelihood, s = -grad(NLL). No online penalty or
retirement kernel is used here. The online handoff resets *all* Adam moments.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.learner import LearningBatch, MemoryLearner, predict_histories
from numerical_experiment.picar.models import batch_digest
from src.continual import picar_math as pm
from src.seeding import derive_component_seed, derive_seed_map


SCHEMA = 1
FIT_POLICY = {
    "kl_max_passes": 12, "nll_max_passes": 16, "min_passes": 6,
    "patience": 4, "plateau_relative_tolerance": 1e-4,
    "batch_size": 1, "block_seconds": 45., "block_items": 32,
    "fisher_draws": 4, "diagnostic_histories": 8,
    "fisher_relative_tolerance": .5, "near_zero": 1e-12,
    "quadratic_scales": [-.1, -.05, -.01, .01, .05, .1],
    "selection": "stagewise minimum fixed-validation teacher_KL then joint_NLL; earliest tie; includes epoch zero",
    "plateau": "training objective fails relative improvement for patience epochs; independent of validation selection",
    "optimizer": "separate LTM-only Adam per offline stage; fresh all-bank Adam at each published fit/handoff",
    "readiness": "measurable accepted train NLL reduction and nonworsening validation; plateau is diagnostic, not a global MLE claim",
}
SEED_NAMES = ("offline_order", "offline_global", "bulk_model_labels",
              "independent_model_labels", "diagnostic_subset")


def _tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _global_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _restore_global(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def _zeros_like(values):
    return {name: torch.zeros(value.shape, dtype=torch.float64) for name, value in values.items()}


def _mean(values, count):
    if count <= 0:
        raise ValueError("a Fisher mean needs a positive history count")
    return {name: value / count for name, value in values.items()}


def _trace(values):
    return sum(float(value.double().sum()) for value in values.values())


def _disagreement(left, right, near_zero):
    absolute = math.sqrt(sum(float((left[n] - right[n]).double().square().sum()) for n in right))
    norm = math.sqrt(sum(float(v.double().square().sum()) for v in right.values()))
    return {"absolute_l2": absolute, "reference_l2": norm,
            "relative_l2": absolute / norm if norm > near_zero else None,
            "near_zero_reference": norm <= near_zero}


class InitializationFit:
    """One resumable fit or Fisher unit. ``s`` is the complete store payload.

    ``run_fit`` ends at fit_done, preserving both distilled_learner and the
    validation-selected NLL learner. A separate unit supplies that state to
    ``run_fisher``. Neither method completes/seals the caller's RunStore.
    """

    def __init__(self, model, config, train_records, validation_records, store, stopped,
                 *, pristine, acquired_count, fitted_state=None):
        self.model, self.config, self.store, self.stopped = model, config, store, stopped
        self.policy = copy.deepcopy(FIT_POLICY)
        if self.policy["fisher_draws"] != 4:
            raise ValueError("Phase3Fix1 fixes the nested model-label budget at four draws")
        self.train = sorted(train_records, key=lambda r: r["record_id"])
        self.validation = sorted(validation_records, key=lambda r: r["record_id"])
        self._validate_data(acquired_count)
        if not math.isclose(config.learning.temperature, 2.8375266913896, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Phase3Fix1 freezes likelihood temperature at 2.8375266913896")
        if config.learning.action_variance != 16.:
            raise ValueError("Phase3Fix1 freezes action variance at 16")
        self.learner = MemoryLearner(model, config.learning, config.seeds(0))
        self.ltm = {name: self.learner.params[name] for name in self.learner.banks["ltm"]}
        self._validate_learner(pristine)
        self._validate_stm(pristine["parameters"])
        self.seeds = derive_seed_map(config.seed, SEED_NAMES)
        self.order_rng = torch.Generator().manual_seed(self.seeds["offline_order"])
        self.identity = {
            "schema_version": SCHEMA, "numerics": "plan19-phase3-fix1-v1",
            "config": config.to_dict(), "policy": self.policy, "seeds": self.seeds,
            "pristine_parameters": _tensor_hash(pristine["parameters"]),
            "data": {"train": self._references(self.train),
                     "validation": self._references(self.validation)},
            "acquired_count": acquired_count,
            "fitted_identity": fitted_state.get("accepted_identity") if fitted_state else None,
        }
        self.optimizer = self._ltm_optimizer()
        self._pending_metrics = []
        self._items = 0
        self._checkpoint_seconds = 0.
        self._last_save = time.monotonic()
        self._started = time.monotonic()
        saved = store.load_checkpoint(trusted=True)
        if saved is not None:
            if canonical_json(saved["identity"]) != canonical_json(self.identity):
                raise ValueError("initialization resume identity differs")
            self.s = saved
            self._validate_learner(saved["learner"])
            self.learner.load_state_dict(saved["learner"])
            self.optimizer.load_state_dict(copy.deepcopy(saved["fit_optimizer"]))
            self.order_rng.set_state(saved["order_rng"].cpu())
            _restore_global(saved["rng"], self.learner.device)
        else:
            if pristine["retired_count"] or pristine["update_count"] or any(
                    bool(v.count_nonzero()) for v in pristine["diagonal"].values()):
                raise ValueError("each data rung requires an unfitted pristine learner")
            self.learner.load_state_dict(pristine)
            self._validate_stm(pristine["parameters"])
            random.seed(self.seeds["offline_global"])
            np.random.seed(self.seeds["offline_global"] % 2**32)
            torch.manual_seed(self.seeds["offline_global"])
            if self.learner.device.type == "cuda":
                torch.cuda.manual_seed(self.seeds["offline_global"])
            self.s = {
                "schema_version": SCHEMA, "identity": self.identity, "stage": "kl",
                "acquired_count": acquired_count, "eligible": [], "retired": [],
                "curves": [], "selection": {}, "accepted_identity": None,
                "distill_receipt": None, "distilled_learner": None,
                "optimization_exposures": {"kl": 0, "nll": 0},
                "optimization_steps": {"kl": 0, "nll": 0},
                "metrics_cursor": 0, "timings": {"active_seconds": 0., "checkpoint_seconds": 0.},
                "selected_exploration_std": config.learning.exploration_std,
                "exploration_calibration": None,
                "distinct_bootstrap_samples": len(self.train),
            }
            self._start_stage()
            if fitted_state is not None:
                self._load_fit(fitted_state)
        self._neutral = {name: pristine["parameters"][name].clone() for name in self.learner.banks["stm"]}
        self._check_stm()
        if "fisher" in self.s:
            self._validate_fisher()
        self.model.distillation_receipt = copy.deepcopy(self.s["distill_receipt"])
        self.model.eval()
        self._base_seconds = self.s["timings"]["active_seconds"]

    def _validate_data(self, acquired):
        if not self.train or not self.validation:
            raise ValueError("initialization needs nonempty training and fixed validation histories")
        records = self.train + self.validation
        ids = [r["record_id"] for r in records]
        arrivals = [r["arrival"] for r in records]
        if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("initialization record IDs must be unique nonempty strings")
        if type(acquired) is not int or acquired < len(records) or any(
                type(a) is not int or not 0 <= a < acquired for a in arrivals):
            raise ValueError("acquired_count must include every training/validation capture")
        if len(arrivals) != len(set(arrivals)):
            raise ValueError("arrival IDs must be globally unique")
        if {r["episode_id"] for r in self.train} & {r["episode_id"] for r in self.validation}:
            raise ValueError("whole-episode training/validation split overlaps")

    @staticmethod
    def _references(records):
        return [{"record_id": r["record_id"], "arrival": r["arrival"], "episode_id": r["episode_id"],
                 "history_sha256": batch_digest(r["history"]), "baseline": float(r["baseline"]),
                 "bin": int(r["bin"]), "targets_sha256": _tensor_hash({
                     "actions": r["logits"], "teacher": r["executed_probabilities"]})}
                for r in records]

    def _validate_stm(self, parameters):
        names = self.learner.banks["stm"]
        for name in names:
            if not name.endswith((".A", ".B")):
                raise ValueError("initial STM must use random-A/zero-B factors")
            value = parameters[name]
            if name.endswith(".B") and bool(value.count_nonzero()):
                raise ValueError("initial STM B must be zero")
            if name.endswith(".A") and not bool(value.count_nonzero()):
                raise ValueError("initial STM A must remain live for later online learning")

    def _validate_learner(self, state):
        if state.get("schema_version") != 1 or state.get("banks") != self.learner.banks:
            raise ValueError("initialization learner schema/parameter banks differ")
        for key, expected in (("parameters", self.learner.params), ("anchor", self.ltm), ("diagonal", self.ltm)):
            if set(state.get(key, {})) != set(expected):
                raise ValueError(f"initialization learner {key} names differ")
            for name, parameter in expected.items():
                value = state[key][name]
                if value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"invalid initialization {key} shape/value")
                if key == "diagonal" and bool((value < 0).any()):
                    raise ValueError("initialization diagonal must be nonnegative")
        for key in ("version", "retired_count", "update_count"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError("invalid initialization learner counter")

    def _validate_fisher(self):
        f = self.s["fisher"]
        expected_subset = [r["record_id"] for r in self._subset(self.train)]
        if (f.get("subset_ids") != expected_subset or type(f.get("count")) is not int
                or not 0 <= f["count"] <= len(self.train)
                or type(f.get("independent_count")) is not int
                or not 0 <= f["independent_count"] <= len(expected_subset)
                or (f["count"] < len(self.train) and f["independent_count"] != 0)):
            raise ValueError("invalid Fisher history/draw cursor or diagnostic subset")
        if f["anchor_sha256"] != _tensor_hash(self._parameters()):
            raise ValueError("bulk Fisher fitted anchor differs")
        groups = list(f["nested_sums"].values()) + [f["subset_sum"], f["independent_sum"]]
        if set(f["nested_sums"]) != {"1", "2", "4"}:
            raise ValueError("missing nested Fisher draw sums")
        for values in groups:
            if set(values) != set(self.ltm):
                raise ValueError("Fisher sum parameter names differ")
            if any(v.shape != self.ltm[n].shape or not bool((torch.isfinite(v) & (v >= 0)).all())
                   for n, v in values.items()):
                raise ValueError("invalid raw Fisher sum shape/value")
        if self.s["stage"] in {"local_quadratic", "done"}:
            if (f["count"] != len(self.train) or f["independent_count"] != len(expected_subset)
                    or self.learner.retired_count != len(self.train)
                    or self.s["retired"] != sorted(r["arrival"] for r in self.train)
                    or self.s["eligible"]):
                raise ValueError("bulk archive count/admission differs from completed Fisher")
            if any(not torch.equal(self.learner.anchor[n], p.detach()) for n, p in self.ltm.items()):
                raise ValueError("bulk archive anchor differs from fitted parameters")

    def _check_stm(self):
        if any(not torch.equal(self.learner.params[n].detach().cpu(), v) for n, v in self._neutral.items()):
            raise ValueError("offline initialization changed STM")

    def _ltm_optimizer(self):
        return torch.optim.Adam(self.ltm.values(), lr=self.config.learning.learning_rate, weight_decay=0.)

    def _reset_online_optimizer(self):
        self.learner.optimizer = torch.optim.Adam(self.learner.params.values(),
            lr=self.config.learning.learning_rate, weight_decay=0.)

    def _parameters(self):
        return {n: p.detach().cpu().clone() for n, p in self.ltm.items()}

    def _copy_parameters(self, values):
        with torch.no_grad():
            for name, parameter in self.ltm.items():
                parameter.copy_(values[name].to(parameter))
        self.learner.version += 1
        self.model.mark_published()

    def _batch(self, records):
        # Offline histories include *all* recorded causal context, even when a
        # context frame is not itself a supervised endpoint in this data rung.
        return LearningBatch([r["history"] for r in records],
            torch.tensor([r["baseline"] for r in records], device=self.learner.device, dtype=self.learner.dtype),
            torch.tensor([r["bin"] for r in records], device=self.learner.device, dtype=torch.long),
            torch.stack([r["logits"].detach().reshape(7).to(self.learner.device, self.learner.dtype) for r in records]),
            torch.tensor([r["arrival"] for r in records], device=self.learner.device))

    def _losses(self, records):
        batch = self._batch(records)
        output = predict_histories(self.model, self.learner.params, batch.histories)
        nll = pm.joint_nll(output, batch.labels, batch.actions, self.learner.design(batch))
        critic = -output["return_logits"].log_softmax(-1).gather(1, batch.labels[:, None]).squeeze(1)
        logits = torch.cat((output["action_mean"], output["action_mean"].new_zeros(len(records), 1)), -1)
        target = torch.stack([r["executed_probabilities"].detach().reshape(8).to(logits) for r in records])
        if not bool((target >= 0).all()) or not torch.allclose(target.sum(-1), target.new_ones(len(records)), atol=1e-6):
            raise ValueError("recorded teacher probabilities must be normalized")
        kl = F.kl_div(logits.log_softmax(-1), target, reduction="none").sum(-1)
        return {"teacher_kl": kl, "joint_nll": nll, "actor_nll": nll - critic, "critic_nll": critic}

    def _start_stage(self):
        self.optimizer = self._ltm_optimizer()
        self.s["epoch"] = 0
        self.s["phase"] = "evaluate"
        self.s["cursor"] = 0
        self.s["order"] = []
        self.s["epoch_diagnostics"] = self._empty_diagnostics()
        self.s["eval_sums"] = {key: {} for key in ("train", "validation")}
        self.s["train_plateau"] = {"best": None, "stale_epochs": 0, "reached": False}
        self.s["best_parameters"] = None

    @staticmethod
    def _empty_diagnostics():
        return {"steps": 0, "gradient_norm_sum": 0., "gradient_norm_max": 0.,
                "update_norm_sum": 0., "update_norm_max": 0., "parameter_norm": 0.}

    def _train_batch(self):
        s, size = self.s, self.policy["batch_size"]
        indices = s["order"][s["cursor"]:s["cursor"] + size]
        records = [self.train[i] for i in indices]
        losses = self._losses(records)
        loss = losses["teacher_kl" if s["stage"] == "kl" else "joint_nll"].mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite initialization objective")
        self.model.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(loss, tuple(self.ltm.values()), allow_unused=True)
        before = {n: p.detach().clone() for n, p in self.ltm.items()}
        gradient_squared = 0.
        for (name, parameter), gradient in zip(self.ltm.items(), gradients):
            if gradient is not None:
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("nonfinite initialization gradient")
                gradient_squared += float(gradient.detach().double().square().sum())
            parameter.grad = gradient
        self.optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in self.ltm.values()):
            raise FloatingPointError("nonfinite initialization parameter")
        update_norm = math.sqrt(sum(float((p.detach() - before[n]).double().square().sum()) for n, p in self.ltm.items()))
        grad_norm = math.sqrt(gradient_squared)
        diagnostics = s["epoch_diagnostics"]
        diagnostics["steps"] += 1
        diagnostics["gradient_norm_sum"] += grad_norm
        diagnostics["gradient_norm_max"] = max(diagnostics["gradient_norm_max"], grad_norm)
        diagnostics["update_norm_sum"] += update_norm
        diagnostics["update_norm_max"] = max(diagnostics["update_norm_max"], update_norm)
        diagnostics["parameter_norm"] = math.sqrt(sum(float(p.detach().double().square().sum()) for p in self.ltm.values()))
        s["cursor"] += len(records)
        s["optimization_exposures"][s["stage"]] += len(records)
        s["optimization_steps"][s["stage"]] += 1
        self.learner.update_count += 1
        self.learner.version += 1
        self.model.mark_published()
        if s["cursor"] == len(self.train):
            s["phase"], s["cursor"] = "evaluate", 0
            s["eval_sums"] = {key: {} for key in ("train", "validation")}

    @torch.no_grad()
    def _evaluate_record(self):
        s = self.s
        cursor = s["cursor"]
        split = "train" if cursor < len(self.train) else "validation"
        record = self.train[cursor] if split == "train" else self.validation[cursor - len(self.train)]
        values = self._losses([record])
        for key, value in values.items():
            scalar = float(value.sum())
            if not math.isfinite(scalar):
                raise FloatingPointError("nonfinite fixed-set diagnostic")
            s["eval_sums"][split][key] = s["eval_sums"][split].get(key, 0.) + scalar
        s["cursor"] += 1
        if s["cursor"] == len(self.train) + len(self.validation):
            self._finish_epoch()

    def _finish_epoch(self):
        s, stage, epoch = self.s, self.s["stage"], self.s["epoch"]
        objective = "teacher_kl" if stage == "kl" else "joint_nll"
        diagnostic = copy.deepcopy(s["epoch_diagnostics"])
        for key in ("gradient_norm", "update_norm"):
            diagnostic[key + "_mean"] = diagnostic[key + "_sum"] / diagnostic["steps"] if diagnostic["steps"] else None
        row = {"stage": stage, "epoch": epoch,
               "optimization_exposures": sum(s["optimization_exposures"].values()),
               "stage_exposures": s["optimization_exposures"][stage],
               "train": {k: v / len(self.train) for k, v in s["eval_sums"]["train"].items()},
               "validation": {k: v / len(self.validation) for k, v in s["eval_sums"]["validation"].items()},
               "diagnostics": diagnostic}
        s["curves"].append(row)
        self._pending_metrics.append({"phase": 3, "condition": "initialization_" + stage,
            "step": len(s["curves"]), "episode": epoch,
            "data_nll": row["train"]["joint_nll"], "total_objective": row["train"][objective],
            "ewc": 0., "ridge": 0., "fd": 0., "replay_exposures": 0,
            "new_transitions": 0, "model_version": self.learner.version})
        value = row["validation"][objective]
        if stage not in s["selection"] or value < s["selection"][stage]["validation_objective"]:
            s["best_parameters"] = self._parameters()
            s["selection"][stage] = {"stage": stage, "epoch": epoch, "objective": objective,
                "validation_objective": value, "train_objective": row["train"][objective],
                "parameters_sha256": _tensor_hash(s["best_parameters"]),
                "selection_rule": self.policy["selection"]}
        plateau = s["train_plateau"]
        train_value = row["train"][objective]
        tolerance = self.policy["plateau_relative_tolerance"] * max(1., abs(plateau["best"] or 0.))
        if plateau["best"] is None or train_value < plateau["best"] - tolerance:
            plateau["best"], plateau["stale_epochs"] = train_value, 0
        elif epoch:
            plateau["stale_epochs"] += 1
        plateau["reached"] = epoch >= self.policy["min_passes"] and plateau["stale_epochs"] >= self.policy["patience"]
        row["train_plateau"] = copy.deepcopy(plateau)
        if epoch >= self.policy[stage + "_max_passes"] or plateau["reached"]:
            s["selection"][stage]["completed_passes"] = epoch
            s["selection"][stage]["train_plateau"] = copy.deepcopy(plateau)
            self._copy_parameters(s["best_parameters"])
            self._reset_online_optimizer()
            if stage == "kl":
                s["distill_receipt"] = {"stage": "one_time_multiepoch_recorded_teacher_kl",
                    "passes": epoch, "optimization_exposures": s["optimization_exposures"]["kl"],
                    "steps": s["optimization_steps"]["kl"], "train_count": len(self.train),
                    "batch_size": self.policy["batch_size"], "selection": copy.deepcopy(s["selection"]["kl"]),
                    "training_records_sha256": hashlib.sha256(canonical_json(self.identity["data"]["train"]).encode()).hexdigest(),
                    "validation_targets_used_for_gradients": False}
                self.model.distillation_receipt = copy.deepcopy(s["distill_receipt"])
                s["distilled_learner"] = self.learner.state_dict()
                s["stage"] = "nll"
                self._start_stage()
            else:
                s["accepted_identity"] = copy.deepcopy(s["selection"]["nll"])
                self.learner.anchor = {n: p.detach().clone() for n, p in self.ltm.items()}
                s["fitted_direction"] = {n: p.detach().cpu() - s["distilled_learner"]["parameters"][n]
                                         for n, p in self.ltm.items()}
                s["stage"] = "fit_done"
                s["best_parameters"] = None
            self.save()
        else:
            s["epoch"] += 1
            s["phase"], s["cursor"] = "train", 0
            s["order"] = torch.randperm(len(self.train), generator=self.order_rng).tolist()
            s["epoch_diagnostics"] = self._empty_diagnostics()
            self.save()

    def save(self):
        self._check_stm()
        self.s["learner"] = self.learner.state_dict()
        self.s["fit_optimizer"] = copy.deepcopy(self.optimizer.state_dict())
        self.s["rng"] = _global_rng(self.learner.device)
        self.s["order_rng"] = self.order_rng.get_state().clone()
        self.s["metrics_cursor"] += len(self._pending_metrics)
        self.s["timings"]["checkpoint_seconds"] += self._checkpoint_seconds
        self.s["timings"]["active_seconds"] = self._base_seconds + time.monotonic() - self._started
        started = time.monotonic()
        self.store.checkpoint(self.s, metrics=self._pending_metrics, progress={
            "schema_version": SCHEMA, "phase": 3, "status": "initialization: " + self.s["stage"],
            "stage": self.s["stage"], "epoch": self.s["epoch"], "cursor": self.s["cursor"],
            "counts": self._progress_counts(), "timings": self.s["timings"],
            "diagnostic_curves": self.s["curves"],
            "fisher_histories": self.s.get("fisher", {}).get("count", 0)})
        self._pending_metrics = []
        # Commit timing from the preceding write in the next block. Do not
        # mutate published state: completed resume must return the same summary.
        # The campaign's outer ledger includes the final write and callback.
        self._checkpoint_seconds = time.monotonic() - started
        self._last_save, self._items = time.monotonic(), 0
        # The callback may raise for STOP/resource admission, always *after*
        # committing parameters, metrics, optimizer and the exact next cursor.
        self.stopped()
        _restore_global(self.s["rng"], self.learner.device)

    def _block(self):
        self._items += 1
        if self._items >= self.policy["block_items"] or time.monotonic() - self._last_save >= self.policy["block_seconds"]:
            self.save()

    def _counts(self):
        return {"train_histories": len(self.train), "validation_histories": len(self.validation),
                "train_episodes": len({r["episode_id"] for r in self.train}),
                "validation_episodes": len({r["episode_id"] for r in self.validation}),
                "acquired_count": self.s["acquired_count"],
                "optimization_exposures": copy.deepcopy(self.s["optimization_exposures"]),
                "optimization_steps": copy.deepcopy(self.s["optimization_steps"]),
                "fisher_histories": self.s.get("fisher", {}).get("count", 0),
                "model_score_draws": self.s.get("fisher", {}).get("count", 0) * self.policy["fisher_draws"],
                "independent_diagnostic_draws": self.s.get("fisher", {}).get("independent_count", 0) * self.policy["fisher_draws"]}

    def _progress_counts(self):
        counts = self._counts()
        for key, suffix in (("optimization_exposures", "exposures"), ("optimization_steps", "steps")):
            stages = counts.pop(key)
            counts[key] = sum(stages.values())
            counts.update({stage + "_" + suffix: value for stage, value in stages.items()})
        counts["step"] = sum(self.s["optimization_steps"].values())
        counts["target"] = len(self.train) * (self.policy["kl_max_passes"] + self.policy["nll_max_passes"])
        return counts

    def _readiness(self):
        curves = [r for r in self.s["curves"] if r["stage"] == "nll"]
        selected = self.s["selection"].get("nll")
        ready, flags = False, []
        if curves and selected and "completed_passes" in selected:
            baseline = curves[0]
            accepted = next(r for r in curves if r["epoch"] == selected["epoch"])
            tolerance = self.policy["plateau_relative_tolerance"] * max(1., abs(baseline["train"]["joint_nll"]))
            ready = (baseline["train"]["joint_nll"] - accepted["train"]["joint_nll"] > tolerance
                     and accepted["validation"]["joint_nll"] <= baseline["validation"]["joint_nll"]
                     and selected["epoch"] > 0 and selected["completed_passes"] >= self.policy["min_passes"])
            if not selected["train_plateau"]["reached"]:
                flags.append("training_plateau_not_established_within_pass_cap")
            if selected["epoch"] < selected["completed_passes"]:
                flags.append("validation_selected_earlier_epoch")
            if accepted["validation"]["teacher_kl"] > 1.1 * max(baseline["validation"]["teacher_kl"], 1e-12):
                flags.append("NLL_fit_increased_validation_teacher_KL; inspect_initial_A_capability")
        if not ready:
            flags.append("accepted_likelihood_optimization_evidence_unresolved")
        return {"ready": ready, "optimization_ready": ready,
                "train_plateau": bool(selected and selected.get("train_plateau", {}).get("reached")),
                "data_adequacy": {"status": "unresolved", "requires": "coverage/data-size curves and separate initial-A capability evaluation"},
                "generalization_established": False, "global_mle_claimed": False,
                "final_B_required": False, "flags": flags}

    def _summary(self):
        return {"schema_version": SCHEMA, "stage": self.s["stage"],
                "curves": copy.deepcopy(self.s["curves"]), "selection": copy.deepcopy(self.s["selection"]),
                "accepted_identity": copy.deepcopy(self.s["accepted_identity"]),
                "distill_receipt": copy.deepcopy(self.s["distill_receipt"]),
                "counts": self._counts(), "readiness": self._readiness(),
                "policy": copy.deepcopy(self.policy), "seeds": dict(self.seeds),
                "timings": copy.deepcopy(self.s["timings"]),
                "initial_pi_used": False, "initial_penalties_used": False,
                "stm_unchanged": True, "conditioning": "fixed recorded full causal histories and recorded baselines"}

    def run_fit(self):
        if self.identity["fitted_identity"] is not None:
            raise ValueError("a Fisher unit cannot restart fitting")
        if self.s["stage"] == "fit_done":
            return self._summary()
        if getattr(self.store, "completed", False):
            raise ValueError("completed run cannot be trained")
        if "learner" not in self.s:
            self.save()
        while self.s["stage"] in {"kl", "nll"}:
            if self.s["phase"] == "train":
                self._train_batch()
            else:
                self._evaluate_record()
            self._block()
        if self.s["stage"] != "fit_done":
            raise ValueError("fit checkpoint is in an incompatible stage")
        return self._summary()

    def _load_fit(self, fitted):
        if (fitted.get("schema_version") != SCHEMA or fitted.get("stage") != "fit_done"
                or not fitted.get("distill_receipt") or not fitted.get("distilled_learner")
                or not fitted.get("accepted_identity")):
            raise ValueError("bulk Fisher requires a completed accepted fit checkpoint")
        for key in ("config", "policy", "pristine_parameters", "data"):
            if canonical_json(fitted["identity"][key]) != canonical_json(self.identity[key]):
                raise ValueError("Fisher fit identity/data/likelihood differs")
        if fitted["acquired_count"] > self.s["acquired_count"]:
            raise ValueError("Fisher acquisition offset cannot omit fitted captures")
        self._validate_learner(fitted["learner"])
        if fitted["learner"]["retired_count"] or fitted["retired"] or fitted["eligible"]:
            raise ValueError("fitted histories must not already have archive/replay admission")
        self.learner.load_state_dict(fitted["learner"])
        if _tensor_hash(self._parameters()) != fitted["accepted_identity"]["parameters_sha256"]:
            raise ValueError("accepted fitted parameters do not match their identity")
        for key in ("curves", "selection", "accepted_identity", "distill_receipt", "distilled_learner",
                    "optimization_exposures", "optimization_steps", "fitted_direction"):
            self.s[key] = copy.deepcopy(fitted[key])
        self.s["stage"] = "fisher"
        zeros = _zeros_like(self.ltm)
        self.s["fisher"] = {"count": 0, "independent_count": 0,
            "nested_sums": {str(m): copy.deepcopy(zeros) for m in (1, 2, 4)},
            "subset_sum": copy.deepcopy(zeros), "independent_sum": copy.deepcopy(zeros),
            "history_estimates": {}, "subset_ids": [r["record_id"] for r in self._subset(self.train)],
            "anchor_sha256": _tensor_hash(self._parameters())}

    def _subset(self, records):
        return sorted(records, key=lambda r: derive_component_seed(self.seeds["diagnostic_subset"], r["record_id"]))[:self.policy["diagnostic_histories"]]

    def _score_draws(self, record, *, independent=False):
        if _tensor_hash(self._parameters()) != self.s["fisher"]["anchor_sha256"]:
            raise ValueError("bulk Fisher anchor moved")
        batch = self._batch([record])
        parameters = {n: p.detach().clone().requires_grad_(True) for n, p in self.ltm.items()}
        output = predict_histories(self.model, parameters, batch.histories)
        design = self.learner.design(batch)
        component = "independent_model_labels" if independent else "bulk_model_labels"
        generator = torch.Generator(device=self.learner.device).manual_seed(
            derive_component_seed(self.seeds[component], record["record_id"]))
        total, nested = _zeros_like(parameters), {}
        for draw in range(1, self.policy["fisher_draws"] + 1):
            labels, actions = pm.sample_model_labels(output, design, generator=generator)
            nll = pm.joint_nll(output, labels.detach(), actions.detach(), design).sum()
            gradients = torch.autograd.grad(nll, tuple(parameters.values()), allow_unused=True,
                                             retain_graph=draw < self.policy["fisher_draws"])
            for (name, parameter), gradient in zip(parameters.items(), gradients):
                # Each gradient is for one model-labelled history. Squaring
                # grad(NLL) equals squaring the score, never a batch-mean OPG.
                gradient = torch.zeros_like(parameter) if gradient is None else gradient.detach()
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("nonfinite model score")
                total[name].add_(gradient.cpu().double().square())
            if draw in (1, 2, 4):
                nested[str(draw)] = _mean(total, draw)
        return nested

    def _fisher_record(self):
        f = self.s["fisher"]
        if f["count"] < len(self.train):
            record = self.train[f["count"]]
            nested = self._score_draws(record)
            for draws, values in nested.items():
                for name, value in values.items():
                    f["nested_sums"][draws][name].add_(value)
            if record["record_id"] in f["subset_ids"]:
                for name, value in nested["4"].items():
                    f["subset_sum"][name].add_(value)
            f["count"] += 1
            if f["count"] in {max(1, len(self.train) // 4), max(1, len(self.train) // 2), len(self.train)}:
                f["history_estimates"][str(f["count"])] = _mean(f["nested_sums"]["4"], f["count"])
        else:
            subset = {r["record_id"]: r for r in self.train}
            record = subset[f["subset_ids"][f["independent_count"]]]
            values = self._score_draws(record, independent=True)["4"]
            for name, value in values.items():
                f["independent_sum"][name].add_(value)
            f["independent_count"] += 1
        if f["count"] == len(self.train) and f["independent_count"] == len(f["subset_ids"]):
            self._finish_fisher()

    def _finish_fisher(self):
        f = self.s["fisher"]
        diagonal = _mean(f["nested_sums"]["4"], len(self.train))
        if any(not bool((torch.isfinite(v) & (v >= 0)).all()) for v in diagonal.values()):
            raise FloatingPointError("invalid raw bulk Fisher; no projection/floor is applied")
        self.learner.diagonal = {n: v.to(self.ltm[n]) for n, v in diagonal.items()}
        if any(not bool(torch.isfinite(v).all()) for v in self.learner.diagonal.values()):
            raise FloatingPointError("bulk Fisher cannot be represented in learner dtype")
        self.learner.anchor = {n: p.detach().clone() for n, p in self.ltm.items()}
        self.learner.retired_count = len(self.train)
        self._reset_online_optimizer()
        self.s["retired"] = sorted(r["arrival"] for r in self.train)
        self.s["eligible"] = []
        self.s["stage"] = "local_quadratic"
        self.s["local"] = {"cursor": 0, "sums": [{"exact_predictive_kl": 0., "observed_nll_change": 0.,
            "first_order": 0., "corrected_observed_nll_change": 0.} for _ in self.policy["quadratic_scales"]]}
        self.save()

    def _quadratic_record(self):
        local, direction = self.s["local"], self.s["fitted_direction"]
        records = self._subset(self.validation)
        record = records[local["cursor"]]
        batch = self._batch([record])
        design = self.learner.design(batch)
        parameters = {n: p.detach().clone().requires_grad_(True) for n, p in self.ltm.items()}
        source = predict_histories(self.model, parameters, batch.histories)
        base_nll = pm.joint_nll(source, batch.labels, batch.actions, design).mean()
        gradients = torch.autograd.grad(base_nll, tuple(parameters.values()), allow_unused=True)
        slope = sum(float((g.detach().double() * direction[n].to(g).double()).sum())
                    for n, g in zip(parameters, gradients) if g is not None)
        source = {k: v.detach() for k, v in source.items()}
        with torch.no_grad():
            for scale, accumulator in zip(self.policy["quadratic_scales"], local["sums"]):
                shifted = {n: p.detach() + scale * direction[n].to(p) for n, p in parameters.items()}
                target = predict_histories(self.model, shifted, batch.histories)
                change = float(pm.joint_nll(target, batch.labels, batch.actions, design).mean() - base_nll.detach())
                values = {"exact_predictive_kl": float(pm.directed_kl(source, target, design).mean()),
                          "observed_nll_change": change, "first_order": scale * slope,
                          "corrected_observed_nll_change": change - scale * slope}
                for key, value in values.items():
                    if not math.isfinite(value):
                        raise FloatingPointError("nonfinite local quadratic diagnostic")
                    accumulator[key] += value
        local["cursor"] += 1
        if local["cursor"] == len(records):
            self.s["stage"] = "done"
            self.save()

    def _fisher_summary(self):
        summary = self._summary()
        f, count = self.s["fisher"], len(self.train)
        diagonal = _mean(f["nested_sums"]["4"], count)
        near_zero = self.policy["near_zero"]
        draws = [{"draws_per_history": m, "history_count": count,
                  "trace": _trace(_mean(f["nested_sums"][str(m)], count)),
                  **_disagreement(_mean(f["nested_sums"][str(m)], count), diagonal, near_zero)} for m in (1, 2, 4)]
        subset_count = len(f["subset_ids"])
        independent = {"history_count": subset_count, "record_ids": f["subset_ids"], "draws_per_history": 4,
            **_disagreement(_mean(f["independent_sum"], subset_count), _mean(f["subset_sum"], subset_count), near_zero),
            "primary_trace": _trace(_mean(f["subset_sum"], subset_count)),
            "independent_trace": _trace(_mean(f["independent_sum"], subset_count))}
        tolerance = self.policy["fisher_relative_tolerance"]
        stability = {"nested_draws": draws, "independent_subset": independent,
            "history_count": [{"history_count": int(n), "trace": _trace(v),
                               **_disagreement(v, diagonal, near_zero)} for n, v in f["history_estimates"].items()],
            "relative_tolerance": tolerance,
            "within_diagnostic_tolerance": all(v is not None and v <= tolerance for v in (
                draws[1]["relative_l2"], independent["relative_l2"])),
            "interpretation": "nested prefixes measure sensitivity, not iid uncertainty; independent labels isolate Monte Carlo variability"}
        quadratic = sum(float((diagonal[n] * d.double().square()).sum()) for n, d in self.s["fitted_direction"].items())
        records = self._subset(self.validation)
        local_rows = []
        for scale, sums in zip(self.policy["quadratic_scales"], self.s["local"]["sums"]):
            row = {"scale": scale, "diagonal_quadratic": .5 * scale**2 * quadratic,
                   **{k: v / len(records) for k, v in sums.items()}}
            row["absolute_KL_diagonal_error"] = abs(row["exact_predictive_kl"] - row["diagonal_quadratic"])
            row["KL_to_diagonal_ratio"] = row["exact_predictive_kl"] / row["diagonal_quadratic"] if row["diagonal_quadratic"] > near_zero else None
            local_rows.append(row)
        summary.update({"fisher_stability": stability,
            "local_quadratic": {"direction": "accepted_NLL_LTM_minus_selected_distilled_LTM; evaluated_at_accepted_anchor",
                "heldout_record_ids": [r["record_id"] for r in records], "rows": local_rows,
                "interpretation": "training-history diagonal versus heldout exact KL; observed NLL is first-order-corrected, not assumed equal to model Fisher; no dense Fisher or LAN claim"},
            "normalization": {"history_count": count, "draws_per_history": 4, "history_weight": 1. / count,
                "stored_diagonal": "D0 = mean_i mean_m score_L_squared",
                "summed_information": "Lambda0 = N * D0", "summed_information_multiplier": count,
                "mean_trace": _trace(diagonal), "summed_information_trace": count * _trace(diagonal),
                "online_objective": "mean_NLL + omega * D0_quadratic; no additional N multiplier",
                "pi": None, "ridge": 0., "projection_or_floor": False},
            "archive": {"retired_count": count, "retired": list(self.s["retired"]),
                "eligible_count": 0, "validation_admitted": 0, "exactly_once": True},
            "parameter_order": [{"name": n, "shape": list(p.shape)} for n, p in self.ltm.items()],
            "fisher_anchor_sha256": f["anchor_sha256"]})
        numerical_ready = f["count"] == count and _trace(diagonal) > near_zero
        summary["readiness"]["fisher_numerically_ready"] = numerical_ready
        summary["readiness"]["fisher_stability_within_tolerance"] = stability["within_diagnostic_tolerance"]
        summary["readiness"]["ready"] &= numerical_ready
        if not stability["within_diagnostic_tolerance"]:
            summary["readiness"]["flags"].append("model_label_MC_stability_unresolved; not_a_diagonal_full_Fisher_test")
        return summary

    def run_fisher(self):
        if self.identity["fitted_identity"] is None:
            raise ValueError("run_fisher requires the separately completed fitted_state")
        if self.s["stage"] == "done":
            return self._fisher_summary()
        if getattr(self.store, "completed", False):
            raise ValueError("completed run cannot construct Fisher")
        if "learner" not in self.s:
            self.save()
        while self.s["stage"] == "fisher":
            self._fisher_record()
            self._block()
        while self.s["stage"] == "local_quadratic":
            self._quadratic_record()
            self._block()
        if self.s["stage"] != "done":
            raise ValueError("Fisher checkpoint is in an incompatible stage")
        return self._fisher_summary()
