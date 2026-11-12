"""Frozen visual judgments and short navigation guidance for Fix 3."""

from __future__ import annotations

import copy
import hashlib
import math
import time

import torch

from .config import canonical_json
from .dense_roles import (CONTROL_PROMPT, CONTACT_REWARD_PROMPT, ConversationalTeacher,
                          _strict_json, parse_reward, prepare_conversation)
from .models import batch_digest
from .role_repair import native_text


CENTERING_GUIDANCE = """When the route is clear, steer toward the red ball and straighten as it
becomes centered. Prefer keeping it roughly centered while approaching.
When an obstacle blocks the route, go around it; temporarily moving the
ball away from the image center is acceptable."""
CENTERING_CONTROL_PROMPT = CONTROL_PROMPT + "\n" + CENTERING_GUIDANCE
CENTERING_REWARD_PROMPT = """Assess the red ball in this robot camera image.
Blue blocks and walls are not the target. Use only the image.
Is a red ball visible? Give its proximity from 0 to 10: 1 for small and far,
increasing toward 10 as it grows nearby. A large nearby ball deserves 10,
even when partly cropped. Give horizontal centering from 0 to 1: 1 within
the central quarter of image width, decreasing toward 0 at either edge.
Small offsets within that central quarter are equally good.
If no red ball is visible, both scores must be zero.
Output schema: one JSON object with exactly three keys:
{"visible": true, "proximity": 1, "centering": 1}.
visible must be boolean; proximity and centering must be finite numbers."""
REWARD_VERSION = "dense-visual-centering-v1"
LOCALIZATION_PROMPT = """Locate the red ball in this image. Blue objects are not the target.
Give the horizontal coordinate of the ball's center as a fraction of image width:
0 is the left edge, 0.5 the middle, and 1 the right edge. Use its actual image
position, including a partially cropped ball. If absent, use null.
Output only JSON with exactly one key: {"center_x": 0.5}."""


def parse_center(text):
    value = _strict_json(text)
    if set(value) != {"center_x"}:
        raise ValueError("expected a horizontal ball center only")
    x = value["center_x"]
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("visible ball requires a finite fractional horizontal coordinate")
    return max(0., min(1., 1. - max(0., abs(x - .5) - .125) / .375)), float(x)


def parse_components(text):
    result = _strict_json(text)
    if set(result) != {"visible", "proximity", "centering"} or type(result["visible"]) is not bool:
        raise ValueError("centering reward requires visible/proximity/centering only")
    for key, limit in (("proximity", 10.), ("centering", 1.)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f"invalid {key} visual judgment")
    if not result["visible"] and (result["proximity"] != 0 or result["centering"] != 0):
        raise ValueError("absent ball must have zero component scores")
    return result


def compose_score(components, pan_rad, *, bonus=.5):
    values = parse_components(canonical_json(components))
    if isinstance(pan_rad, bool) or not isinstance(pan_rad, (int, float)) or not math.isfinite(pan_rad):
        raise ValueError("measured finite camera pan required")
    if bonus != .5:
        raise ValueError("this reward version fixes the centering bonus at .5")
    weight = max(0., 1. - abs(pan_rad) / math.radians(10))
    increment = bonus * weight * values["centering"] if values["visible"] else 0.
    score = min(10., values["proximity"] + increment) if values["visible"] else 0.
    return {**values, "pan_rad": pan_rad, "pan_weight": weight,
            "centering_bonus": score - values["proximity"], "score": score}


class CenteringTeacher(ConversationalTeacher):
    def __init__(self, encoder, *, variant="centering"):
        if variant not in {"ported", "centering"}:
            raise ValueError("Fix 3 supports ported or centering guidance")
        super().__init__(encoder, variant="ported")
        self.variant = variant
        self.prompt = CONTROL_PROMPT if variant == "ported" else CENTERING_CONTROL_PROMPT
        self.provenance = {**self.provenance, "variant": variant,
            "kind": "fix3-conversational-teacher-v1", "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest()}


class CenteringReward:
    """One frozen RGB judgment; pan enters only the declared bonus arithmetic."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.prompt = CONTACT_REWARD_PROMPT
        self.provenance = {"kind": REWARD_VERSION, "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "localization_prompt": LOCALIZATION_PROMPT,
            "localization_prompt_sha256": hashlib.sha256(LOCALIZATION_PROMPT.encode()).hexdigest(),
            "interface": "separate-proximity-and-localization-v2", "calls_per_visible_image": 2,
            "role": "evaluator", "max_pixels": 76800, "frames": 1, "bonus": .5,
            "pan_cutoff_rad": math.radians(10), "input_whitelist": ["RGB"]}
        self.cache = {}
        self.last_diagnostic = {}

    def state_dict(self):
        return {"provenance": self.provenance, "cache": copy.deepcopy(self.cache)}

    def load_state_dict(self, value):
        if value["provenance"] != self.provenance:
            raise ValueError("frozen evaluator identity changed")
        self.cache = copy.deepcopy(value["cache"])

    @torch.no_grad()
    def score(self, history, pan_rad):
        if history["images"].shape[0] != 1:
            raise ValueError("Fix 3 scorer expects one observed image")
        # prepare_conversation's visual_only branch sends no task or geometry.
        # Context is in this conservative cache key, though not in the prompt.
        key = hashlib.sha256((batch_digest(history) + canonical_json(self.provenance)).encode()).hexdigest()
        started = time.monotonic()
        hit = key in self.cache
        if not hit:
            attempts = []
            with self.encoder._lock:
                values = []
                for prompt, parser in ((self.prompt, parse_reward), (LOCALIZATION_PROMPT, parse_center)):
                    if values and values[0] == 0:
                        values.append((0., None))
                        break
                    inputs = prepare_conversation(self.encoder, history, prompt, visual_only=True)
                    for retry in range(2):
                        raw = native_text(self.encoder, inputs, role="evaluator", max_new_tokens=64)[0]
                        attempts.append(raw)
                        try:
                            parsed = parser(raw)
                        except ValueError:
                            if retry:
                                raise
                        else:
                            values.append(parsed)
                            break
                proximity, (centering, center_x) = values
                components = {"visible": proximity > 0, "proximity": proximity, "centering": centering}
                self.cache[key] = {"components": components, "center_x": center_x, "raw_text": attempts}
        result = compose_score(self.cache[key]["components"], float(pan_rad))
        self.last_diagnostic = {"cache_key": key, "cache_hit": hit,
            "seconds": time.monotonic() - started, **self.cache[key]}
        return {**result, "diagnostic": self.last_diagnostic}


def centering_resolution_audit(temperature=5.):
    """Check quantization of a fixed small return shift across bin alignments."""
    from src.continual import picar_math as pm
    elapsed = torch.ones(20, dtype=torch.float64)
    terminal = torch.zeros(20, dtype=torch.bool)
    terminal[-1] = True
    bonuses = torch.zeros(20, dtype=torch.float64)
    bonuses[-1] = -1.
    returns = []
    for score in (1., 1.5):
        rewards = pm.dense_visual_rewards(elapsed, torch.full_like(elapsed, score), bonuses, terminal)
        discounts = pm.time_discounts(elapsed, .99, .5)
        returns.append(float(pm.discounted_returns(rewards, discounts, terminal)[0]))
    delta = returns[1] - returns[0]
    rows = []
    for bins in (41, 81, 161):
        edges = torch.linspace(-6., 12., bins + 1, dtype=torch.float64)
        # Multiple offsets describe bin alignment, not independent trajectories.
        base = torch.linspace(-4., 9., 1001, dtype=torch.float64)
        labels = pm.bin_returns(base, edges)
        shifted = pm.bin_returns(base + delta, edges)
        fraction = float((labels != shifted).double().mean())
        values = pm.return_bin_values(edges)
        log_weight_change = float((values[shifted] - values[labels]).abs().mean() / temperature)
        rows.append({"bins": bins, "fraction_different_labels": fraction,
            "mean_absolute_log_weight_change": log_weight_change,
            "half_bin_log_weight_bound": float((edges[1] - edges[0]) / (2 * temperature))})
    selected = next((r["bins"] for r in rows if r["fraction_different_labels"] >= .5), 161)
    return {"rows": rows, "selected_bins": selected, "return_difference": delta,
        "unbinned_returns": returns, "temperature": temperature,
        "interpretation": "Constructed alignment diagnostic, not observed learning or independent samples."}
