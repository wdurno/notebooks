"""Recorded-data distillation and equal-weight initialization without online decay."""

import copy
import hashlib

import torch
from torch.nn import functional as F

from numerical_experiment.picar.models import batch_digest
from numerical_experiment.picar_head_awac.data import unpack_history
from numerical_experiment.picar_head_awac.initialization import fit as fit_critics
from numerical_experiment.picar_head_awac.learner import Learner
from numerical_experiment.picar_head_awac.model import Heads, FrozenFeatures
from numerical_experiment.picar_head_rank.numerics import _initial_factor, block_diagnostics
from .config import seed


def representation_parameters(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if not n.startswith("encoder.")}


def restore_representation(model, parameters):
    actual = dict(model.named_parameters())
    if set(parameters) != {n for n in actual if not n.startswith("encoder.")}:
        raise ValueError("initial representation parameter inventory differs")
    with torch.no_grad():
        for name, value in parameters.items():
            actual[name].copy_(value.to(actual[name]))


def validation_episode(episode_id):
    value = hashlib.sha256(f"{seed('initial_split')}:{episode_id}".encode()).digest()
    return int.from_bytes(value[:8], "big") % 5 == 0


def prepare_distillation(model, collection, collection_store, store, check):
    """Cache only frozen transformer outputs; projector and recurrence remain live."""
    state = store.load_checkpoint(trusted=True) or {"records": [], "cursor": 0}
    source = collection["initial_records"]
    for record in source[state["cursor"]:]:
        history = unpack_history(record["history"], collection_store.frames)
        with torch.no_grad():
            semantic, _ = model.encoder.inference_features(history, role="learner")
        state["records"].append({**{k: copy.deepcopy(v) for k, v in record.items()
                                  if k not in {"feature", "next_feature"}},
            "semantic": semantic.detach().cpu(),
            "controls": history["executed_actions"].cpu(), "elapsed": history["elapsed"].cpu(),
            "history_digest": batch_digest(history),
            "validation": validation_episode(record["episode_id"])})
        state["cursor"] += 1
        store.checkpoint(state, progress={"stage": "frozen transformer features", "n": state["cursor"],
                                          "total": len(source)})
        check()
    if not any(r["validation"] for r in state["records"]) or all(r["validation"] for r in state["records"]):
        raise ValueError("whole-episode split has an empty arm")
    return state


def _output(model, record):
    history = {"executed_actions": record["controls"], "elapsed": record["elapsed"]}
    return model._run(history, record["semantic"])[0]["action_mean"]


def distill(model, records, pristine, store, check):
    """Fit only LTM projector, LSTM and action head to saved teacher probabilities."""
    restore_representation(model, pristine)
    model.requires_grad_(False).eval()
    parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith(("projection.", "recurrent.", "action_head.")) and ".ltm." in name:
            parameter.requires_grad_(True)
            parameters.append(parameter)
    optimizer = torch.optim.Adam(parameters, lr=.001)
    state = store.load_checkpoint(trusted=True)
    if state:
        restore_representation(model, state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
    else:
        state = {"epoch": 0, "cursor": 0, "order": None, "curves": [], "best": None,
                 "exposures": 0, "steps": 0,
                 "rng": torch.Generator().manual_seed(seed("distillation_order")).get_state()}
    train = [r for r in records if not r["validation"]]
    valid = [r for r in records if r["validation"]]
    if not train or not valid:
        raise ValueError("whole-episode train/validation records required")
    neutral = {n: v.clone() for n, v in pristine.items() if ".stm." in n}

    def loss(record):
        mean = _output(model, record)
        logq = torch.cat((mean, mean.new_zeros((1, 1))), -1).log_softmax(-1)
        probabilities = record["teacher_probabilities"].reshape(1, 8).to(logq)
        return F.kl_div(logq, probabilities, reduction="batchmean")

    def assessment(split):
        values, turns, matches, turn_matches = [], [], [], []
        with torch.no_grad():
            for record in split:
                value = float(loss(record))
                choice = int(record["teacher_probabilities"].argmax())
                predicted = torch.cat((_output(model, record)[0], torch.zeros(1))).argmax().item()
                values.append(value)
                matches.append(predicted == choice)
                if choice in (0, 1):
                    turns.append(value)
                    turn_matches.append(predicted == choice)
        mean = lambda x: sum(x) / len(x) if x else None
        return {"kl": mean(values), "turn_kl": mean(turns), "agreement": mean(matches),
                "turn_agreement": mean(turn_matches), "turn_count": len(turns)}

    def curve():
        row = {"epoch": state["epoch"], "exposures": state["exposures"],
               **{"train_" + k: v for k, v in assessment(train).items()},
               **{"validation_" + k: v for k, v in assessment(valid).items()}}
        state["curves"].append(row)
        if state["best"] is None or row["validation_kl"] < state["best"]["kl"]:
            state["best"] = {"epoch": state["epoch"], "kl": row["validation_kl"],
                             "parameters": representation_parameters(model)}

    def save():
        state.update(parameters=representation_parameters(model), optimizer=copy.deepcopy(optimizer.state_dict()))
        store.checkpoint(state, progress={"stage": "projector/LSTM KL distillation", "epoch": state["epoch"],
            "cursor": state["cursor"], "total": 12, "exposures": state["exposures"], "curves": state["curves"]})
        check()

    if not state["curves"]:
        curve()
        save()
    while state["epoch"] < 12:
        if state["order"] is None:
            generator = torch.Generator()
            generator.set_state(state["rng"])
            state["order"] = torch.randperm(len(train), generator=generator).tolist()
            state["rng"] = generator.get_state()
        while state["cursor"] < len(train):
            indices = state["order"][state["cursor"]:state["cursor"] + 16]
            optimizer.zero_grad(set_to_none=True)
            value = torch.stack([loss(train[i]) for i in indices]).mean()
            if not torch.isfinite(value):
                raise FloatingPointError("nonfinite distillation loss")
            value.backward()
            optimizer.step()
            state["cursor"] += len(indices)
            state["exposures"] += len(indices)
            state["steps"] += 1
            save()
        state["epoch"] += 1
        state.update(cursor=0, order=None)
        curve()
        save()
        if state["epoch"] >= 6 and state["epoch"] - state["best"]["epoch"] >= 4:
            break
    restore_representation(model, state["best"]["parameters"])
    model.requires_grad_(False)
    if any(not torch.equal(representation_parameters(model)[n], v) for n, v in neutral.items()):
        raise ValueError("STM changed during LTM-only distillation")
    state["selected_parameters"] = representation_parameters(model)
    store.checkpoint(state, progress={"stage": "distillation complete", "epoch": state["epoch"],
                                     "curves": state["curves"]})
    return {"curves": state["curves"], "selected_epoch": state["best"]["epoch"],
            "exposures": state["exposures"], "train_count": len(train), "validation_count": len(valid),
            "initial_pi_used": False, "stm_neutral": True}


def prepare_heads(model, records, collection_store, store, check, identity):
    features = FrozenFeatures(model, identity)
    state = store.load_checkpoint(trusted=True) or {"train": [], "validation": [], "cursor": 0}
    for record in records[state["cursor"]:]:
        current = unpack_history(record["history"], collection_store.frames)
        nxt = unpack_history(record["next_history"], collection_store.frames)
        result = {k: copy.deepcopy(v) for k, v in record.items()
                  if k in {"id", "episode_id", "reward", "return", "discount", "action", "arrival", "visits"}}
        result.update(feature=features(current), next_feature=features(nxt))
        split = "validation" if validation_episode(record["episode_id"]) else "train"
        state[split].append(result)
        state["cursor"] += 1
        state["source_parameters"] = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                                      if n.startswith("action_head.")}
        store.checkpoint(state, progress={"stage": "initialized recurrent features", "n": state["cursor"],
                                          "total": len(records)})
        check()
    if "language_probe" not in state:
        from numerical_experiment.picar.fix2_data import TASK
        from src.robotics.actions import Calibration
        probe_rows = []
        for record in records[:3]:
            history = unpack_history(record["history"], collection_store.frames)
            now = float(history["decision_time"][0, -1])
            variants = {}
            for name, text in (("none", None), ("left", "Steer left while driving forward."),
                               ("right", "Steer right while driving forward.")):
                counterfactual = copy.deepcopy(history)
                counterfactual["contexts"] = [[{"text": TASK, "issued_at": 0.}]]
                counterfactual["status_events"] = [[]]
                if text:
                    counterfactual["contexts"][0].append({"text": text, "issued_at": now})
                with torch.no_grad():
                    feature = features(counterfactual)
                    mean = model.action_head(feature[None])[0].cpu()
                    probabilities = torch.cat((mean, torch.zeros(1))).softmax(-1)
                variants[name] = {"feature": feature, "mean": mean,
                    "steering_degrees": float(Calibration().map_distribution(probabilities.tolist()).steering_rad * 180 / torch.pi)}
            probe_rows.append({"id": record["id"], "same_image_and_action_history": True,
                "left_feature_change": float((variants["left"]["feature"] - variants["none"]["feature"]).norm()),
                "right_feature_change": float((variants["right"]["feature"] - variants["none"]["feature"]).norm()),
                **{key + "_steering_degrees": value["steering_degrees"] for key, value in variants.items()}})
        state["language_probe"] = probe_rows
        store.checkpoint(state, progress={"stage": "counterfactual language-path probe", "n": len(records),
                                          "total": len(records), "language_probe": probe_rows})
        check()
    return {"train_count": len(state["train"]), "validation_count": len(state["validation"]),
            "language_probe": state["language_probe"], "probe_observations_added_to_learning": 0}


def initial_fisher(fitted, records, recipe, store, check):
    heads = Heads().cpu()
    heads.load_state_dict(fitted["selected_heads"])
    learner = Learner(heads, recipe, seed("initial_fisher"), "td_fd")
    state = store.load_checkpoint(trusted=True) or {
        "cursor": 0, "diagonal_sum": torch.zeros(len(learner.ltm)), "factor_chunks": [], "checks": []}
    if "learner" in state:
        learner.load_state_dict(state["learner"])
    while state["cursor"] < len(records):
        i = state["cursor"]
        factor = _initial_factor(learner, records[i])
        state["diagonal_sum"] += factor.square().sum(1)
        state["factor_chunks"].append(factor.cpu())
        learner.archive.update(factor, alpha=i / (i + 1), beta=1 / (i + 1))
        state["cursor"] += 1
        if state["cursor"] % 16 == 0 or state["cursor"] == len(records):
            state["learner"] = learner.state_dict()
            store.checkpoint(state, progress={"stage": "equal-weight rank-64 Fisher", "n": state["cursor"],
                "total": len(records), "archive": learner.archive.diagnostics()})
            check()
    if not state["checks"]:
        for repeat in range(2):
            diagonal = torch.zeros_like(state["diagonal_sum"])
            for record in records[:min(16, len(records))]:
                diagonal += _initial_factor(learner, record).square().sum(1)
            diagonal /= min(16, len(records))
            state["checks"].append({"repeat": repeat, "trace": float(diagonal.sum()),
                                     "sample_count": min(16, len(records))})
    state.update(learner=learner.state_dict(), training_ids=[r["id"] for r in records],
                 raw_sum_count=len(records), initial_pi_used=False)
    store.checkpoint(state, progress={"stage": "initial Fisher complete", "n": len(records),
                                     "total": len(records), "archive": learner.archive.diagnostics()})
    relative = (learner.archive.diagonal - state["diagonal_sum"] / len(records)).norm() / (
        state["diagonal_sum"].norm() / len(records)).clamp_min(1e-12)
    return {"cursor": len(records), "raw_sum_count": len(records), "initial_pi_used": False,
            "archive": learner.archive.diagnostics(), "allocation": block_diagnostics(learner.archive),
            "diagonal_relative_error": float(relative), "integration_checks": state["checks"]}
