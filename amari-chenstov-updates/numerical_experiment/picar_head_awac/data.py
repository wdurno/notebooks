"""Recover immutable initial data and encode exact frozen representations."""

import copy
from pathlib import Path
import torch

from numerical_experiment.picar_phase2.campaign import load_completed_state, read_json
from numerical_experiment.picar_reward_repair.data import restore_stationary, relabel
from numerical_experiment.picar_reward_repair.config import DEFAULT_REFERENCES
from numerical_experiment.picar.models import batch_digest
from numerical_experiment.picar.fix2_data import history_batch


def source_data(source):
    source = Path(source)
    initial_source = Path(read_json(source/"orchestration/config.json")["source_campaign"])
    summary = read_json(initial_source/"orchestration/summary.json")
    fit_path = Path(summary["selected_fit"])
    dataset = restore_stationary({"root": str(initial_source), "fit": str(fit_path)})
    scores_path = next((source/"units").glob("scoring-*/summary.json"))
    scores = read_json(scores_path)["scores"]
    labeled = relabel(dataset, scores, "width6_signed_square", torch.linspace(-6., 12., 162))
    return dataset, labeled


def apply_parameters(model, state):
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["parameters"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError("frozen initializer coordinates differ: "+name)
            parameters[name].copy_(value.to(parameters[name]))
    model.encoder.clear_caches()


def append_history(history, post, context_length=4):
    rows = []
    for j in range(history["images"].shape[1]):
        rows.append({"rgb": history["images"][0,j].mul(255).round().byte().permute(1,2,0).numpy(),
                     "time": float(history["decision_time"][0,j]),
                     "elapsed": float(history["elapsed"][0,j,0]),
                     "action": history["executed_actions"][0,j].tolist()})
    rows = [r for r in rows if r["time"] < post["time"]]
    rows.append(post)
    return history_batch(rows, contexts=history["contexts"][0],
        statuses=history.get("status_events", [[]])[0], deadline=float(history["episode_deadline"][0]),
        length=context_length)


def feature_record(record, feature, *, next_feature=None, reward=0., discount=0.):
    return {"id": record.get("record_id", str(record["arrival"])), "feature": feature.cpu(),
            "next_feature": feature.cpu() if next_feature is None else next_feature.cpu(),
            "action": torch.as_tensor(record["logits"]).float(),
            "return": torch.tensor(record["return"]), "reward": torch.tensor(reward),
            "discount": torch.tensor(discount), "arrival": record["arrival"],
            "episode_id": record["episode_id"], "history_digest": batch_digest(record["history"]),
            "visits": 0}


def pack_history(history, frames):
    packed = {k: copy.deepcopy(v) for k, v in history.items() if k != "images"}
    packed["frame_refs"] = [frames.put(x.mul(255).round().byte().permute(1,2,0).numpy())
                            for x in history["images"][0]]
    return packed


def unpack_history(packed, frames):
    return {**{k: copy.deepcopy(v) for k, v in packed.items() if k != "frame_refs"},
            "images": torch.stack([torch.from_numpy(frames.get(ref).copy()).permute(2,0,1).float()/255
                                   for ref in packed["frame_refs"]])[None]}
