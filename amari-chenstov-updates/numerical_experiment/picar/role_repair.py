"""Versioned Phase 1 role interfaces. No privileged inputs or training here."""

from __future__ import annotations

from contextlib import contextmanager
import json

import torch
from torch import nn
from torch.nn import functional as F

from .models import QwenContextEncoder, _role_scope, render_context, validate_batch
from .teacher import prepare_teacher_distribution


VERSION = "grounded-roles-v1"
ACTION_NAMES = ("drive-left", "drive-right", "drive-forward", "drive-backward",
                "look-left", "look-right", "look-up", "stop")
GROUNDING = """You control the camera of a small wheeled car, looking forward from the car.
The red ball is the target. The blue block is an obstacle, not the target.
Navigate past the obstacle without touching it, approach the ball, then STOP.
Active instructions remain active; completed instructions are history.
Recorded executed_actions are [camera pan radians, camera tilt radians,
steering radians, drive metres/second]. Pan left and tilt up are positive;
steering right is positive. Camera directions are absolute, not increments.
Actions have these physical effects:
0 drive-left: drive forward and steer left; camera centred.
1 drive-right: drive forward and steer right; camera centred.
2 drive-forward: drive straight forward; camera centred.
3 drive-backward: reverse straight; camera centred.
4 look-left: stop drive, release steering, point camera left.
5 look-right: stop drive, release steering, point camera right.
6 look-up: stop drive, release steering, point camera up.
7 stop: stop drive, release steering, centre camera looking forward.
Choose 7 when instructed to stop or when the target is safely approached.
Choose a clear route around the block, not straight into it.
"""
EVALUATOR_PROMPT = """Assess visible progress on the active task, not whether a ball exists.
The goal is to pass the blue obstacle safely, approach the red ball, and stop
with space between the car and ball. Far away is NOT completion. Touching or
facing into the blue obstacle is NOT completion. The ball should be close
and ahead, with no obstacle blocking the path. A sequence of images and
recorded stopped controls can support, but cannot prove, that the car stopped.
Reply with exactly one digit: 1 for apparent successful completion; 0 otherwise.
"""
GROUNDING_V2 = GROUNDING.replace(
    "You control the camera of a small wheeled car, looking forward from the car.",
    "You control a small wheeled car. Images come from its forward-facing camera.").replace(
    "Navigate past the obstacle without touching it, approach the ball, then STOP.",
    "Follow the current user instruction. When navigating, avoid contact with obstacles.")


def prepare_role(encoder, batch, prompt, *, frames=1, max_pixels=None, active_last=False):
    """B independent histories, each with at most `frames` causal RGB images."""
    b, t = validate_batch(batch, encoder.context_length)
    if type(frames) is not int or not 1 <= frames <= encoder.context_length:
        raise ValueError("frames must fit the retained context")
    if max_pixels is not None and (type(max_pixels) is not int or max_pixels < 56 * 56):
        raise ValueError("max_pixels must be at least 56x56")
    texts, images = [], []
    for i in range(b):
        content = []
        times = batch["elapsed"][i, :, 0].cumsum(0)
        if "start_time" in batch:
            times = times + batch["start_time"][i]
        for j in range(max(0, t - frames), t):
            if frames > 1:
                content.append({"type": "text", "text": f"Recorded image at {float(times[j]):.3f} seconds:"})
            content.append({"type": "image"})
            images.append(batch["images"][i, j].detach().cpu())
        context = render_context(batch, i, t - 1, max_chars=encoder.max_context_chars)
        text = context + "\n" + prompt
        if active_last:
            active = [event["text"] for event in json.loads(context)["instructions"]
                      if event["status"] == "active"]
            text += "\nCURRENT USER INSTRUCTIONS:\n" + "\n".join(active)
            text += "\nThe latest instruction overrides conflicting older requests. Give the requested answer now."
        content.append({"type": "text", "text": text})
        texts.append(encoder.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
    options = {} if max_pixels is None else {"max_pixels": max_pixels}
    inputs = dict(encoder.processor(text=texts, images=images, padding=True,
                                  return_tensors="pt", do_rescale=False, **options))
    if inputs["input_ids"].shape[1] > encoder.max_tokens:
        raise ValueError("role context exceeds max_tokens; no silent truncation")
    return inputs


def last_features(hidden, inputs):
    mask = inputs["attention_mask"].to(hidden.device)
    index = (torch.arange(mask.shape[1], device=hidden.device)[None] * mask).max(-1).values
    return hidden[torch.arange(mask.shape[0], device=hidden.device), index]


def vocabulary_scores(encoder, features, labels):
    """FP32 readout; report full-vocabulary mass as well as restricted softmax."""
    encoded = [encoder.processor.tokenizer.encode(x, add_special_tokens=False) for x in labels]
    if any(len(x) != 1 for x in encoded) or len({x[0] for x in encoded}) != len(labels):
        raise ValueError("role answers must be distinct single tokens")
    head = encoder.qwen.lm_head
    x = features.float().to(head.weight.device)
    ids = torch.tensor([x[0] for x in encoded], device=x.device)
    logits = F.linear(x, head.weight[ids].float(),
                      None if head.bias is None else head.bias[ids].float())
    normalizer = None
    # Avoid a full FP32 vocabulary-weight copy just to diagnose answer mass.
    for start in range(0, head.weight.shape[0], 4096):
        block = F.linear(x, head.weight[start:start + 4096].float(),
            None if head.bias is None else head.bias[start:start + 4096].float())
        value = block.logsumexp(-1)
        normalizer = value if normalizer is None else torch.logaddexp(normalizer, value)
    mass = (logits.logsumexp(-1) - normalizer).exp()
    return logits.softmax(-1), {"logits": logits.cpu().tolist(), "allowed_answer_mass": mass.cpu().tolist(),
                                "head_precision": "float32"}


@contextmanager
def _native_policy(encoder, role, *, positions=None):
    """Temporary native-path hooks; no parameter mutation, no shared KV state."""
    core = encoder.qwen.model
    text = core.language_model
    def top_input(module, args, kwargs):
        value = args[0].float()
        if positions is not None:
            kwargs["position_embeddings"] = text.rotary_emb(value, positions)
        else:
            kwargs["position_embeddings"] = tuple(x.float() for x in kwargs["position_embeddings"])
        return (value, *args[1:]), kwargs
    with encoder._lock, _role_scope(role), torch.no_grad(), torch.autocast(
            device_type=text.embed_tokens.weight.device.type, enabled=False):
        prior_rope = getattr(core, "rope_deltas", None)
        handles = []
        try:
            for index in encoder.selected_layers:
                handles.append(text.layers[index].register_forward_pre_hook(top_input, with_kwargs=True))
            core.rope_deltas = None
            yield
        finally:
            for handle in handles:
                handle.remove()
            core.rope_deltas = prior_rope


@torch.no_grad()
def native_hidden(encoder, inputs, *, role="teacher"):
    """Upstream multimodal forward under the split encoder's FP32 suffix policy."""
    device = encoder.qwen.model.language_model.embed_tokens.weight.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    positions, _ = encoder.qwen.model.get_rope_index(inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"), attention_mask=inputs["attention_mask"])
    with _native_policy(encoder, role, positions=positions):
        return encoder.qwen.model(**inputs, use_cache=False, return_dict=True).last_hidden_state


class _FloatHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, hidden):
        return torch.cat([F.linear(hidden.float(), self.head.weight[i:i + 4096].float(),
            None if self.head.bias is None else self.head.bias[i:i + 4096].float())
            for i in range(0, self.head.weight.shape[0], 4096)], -1)


@torch.no_grad()
def native_text(encoder, inputs, *, role="teacher", max_new_tokens=48):
    """Fresh native generation; its local KV cache cannot enter another role."""
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("bounded generation required")
    device = encoder.qwen.model.language_model.embed_tokens.weight.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with _native_policy(encoder, role):
        original = encoder.qwen.lm_head
        try:
            encoder.qwen.lm_head = _FloatHead(original)
            outputs = encoder.qwen.generate(**inputs, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True, logits_to_keep=1,
                pad_token_id=encoder.processor.tokenizer.pad_token_id)
        finally:
            encoder.qwen.lm_head = original
    return encoder.processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)


def parse_action(text):
    """Strict JSON only; a parse failure is not a fabricated stop action."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("teacher did not return valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"action"} or value["action"] not in ACTION_NAMES:
        raise ValueError("teacher must return exactly one known action")
    return ACTION_NAMES.index(value["action"])


class GroundedTeacher:
    def __init__(self, encoder, *, mode="restricted", frames=1, max_pixels=19200, prompt_version=1):
        if not isinstance(encoder, QwenContextEncoder) or mode not in ("restricted", "generated"):
            raise ValueError("real Qwen and a declared grounded teacher mode required")
        self.encoder, self.mode, self.frames, self.max_pixels = encoder, mode, frames, max_pixels
        if prompt_version not in (1, 2):
            raise ValueError("unknown grounded prompt version")
        self.prompt_version = prompt_version
        self.prompt = (GROUNDING if prompt_version == 1 else GROUNDING_V2) + ("Reply with exactly one digit 0 through 7, no explanation." if mode == "restricted"
            else 'Reply with only JSON: {"action":"NAME"}. NAME must be one of: ' + ", ".join(ACTION_NAMES))
        self.provenance = {"kind": f"grounded-roles-v{prompt_version}", "role": "teacher", "mode": mode,
            "frames": frames, "max_pixels": max_pixels, "prompt": self.prompt,
            "prompt_version": prompt_version,
            "target_kind": "soft" if mode == "restricted" else "hard_engineered",
            "calibrated_uncertainty": False, "control_map": "unchanged_probability_weighted"}
        self.last_diagnostic = {}

    @torch.no_grad()
    def distribution(self, batch, *, probability_floor):
        with self.encoder._lock:
            inputs = prepare_role(self.encoder, batch, self.prompt,
                                  frames=self.frames, max_pixels=self.max_pixels, active_last=self.prompt_version == 2)
            if self.mode == "restricted":
                hidden = self.encoder.encode_processed(inputs, role="teacher")
                probabilities, self.last_diagnostic = vocabulary_scores(
                    self.encoder, last_features(hidden, inputs), tuple(map(str, range(8))))
            else:
                texts = native_text(self.encoder, inputs, max_new_tokens=32)
                self.last_diagnostic = {"generated_text": texts}
                choices = [parse_action(text) for text in texts]
                probabilities = F.one_hot(torch.tensor(choices), 8).float()
                self.last_diagnostic = {"generated_text": texts, "generated_choices": choices}
            return prepare_teacher_distribution(probabilities, probability_floor=probability_floor,
                target_kind=self.provenance["target_kind"])


class GroundedEvaluator:
    def __init__(self, encoder, *, frames=3, max_pixels=76800):
        if not isinstance(encoder, QwenContextEncoder):
            raise TypeError("real Qwen required")
        self.encoder, self.frames, self.max_pixels = encoder, frames, max_pixels
        self.provenance = {"kind": VERSION, "role": "evaluator", "frames": frames,
            "max_pixels": max_pixels, "prompt": EVALUATOR_PROMPT, "calibrated": False}
        self.last_diagnostic = {}

    @torch.no_grad()
    def evaluate(self, batch):
        with self.encoder._lock:
            inputs = prepare_role(self.encoder, batch, EVALUATOR_PROMPT,
                                  frames=self.frames, max_pixels=self.max_pixels)
            hidden = self.encoder.encode_processed(inputs, role="evaluator")
            probabilities, self.last_diagnostic = vocabulary_scores(
                self.encoder, last_features(hidden, inputs), ("0", "1"))
            return probabilities[:, 1]
