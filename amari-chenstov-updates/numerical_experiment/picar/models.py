"""Plan 19 model plumbing. Importing this module never loads a checkpoint.

``forward(batch)`` returns return_logits[B,K] and action_mean[B,7]. A batch
contains images[B,T,3,H,W] in [0,1], executed_actions[B,T,4], elapsed[B,T,1]
in seconds, and contexts (B sequences of LanguageEvent or equivalent mappings).
Event times are absolute; start_time[B] (default zero) is the time immediately
before the first retained frame. Every timestep conditions only on its past.
Replay always reconstructs recurrence from the complete bounded raw history.
The action readout is a Gaussian mean, not eight categorical action logits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.seeding import derive_component_seed


MODEL_SCHEMA_VERSION = 1
CONTEXT_SCHEMA_VERSION = 1
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
QWEN_CACHE_DIR = str(Path(__file__).resolve().parents[2] / "cache/numerical_experiment/picar/models")
ADAPTER_TARGETS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
ROLES = ("learner", "teacher", "evaluator")
_ROLE: ContextVar[str | None] = ContextVar("picar_model_role", default=None)


@contextmanager
def _role_scope(role: str):
    if role not in ROLES:
        raise ValueError(f"unknown model role: {role}")
    token = _ROLE.set(role)
    try:
        yield
    finally:
        _ROLE.reset(token)


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class LowRankBank(nn.Module):
    """Fixed scale times B A, with random A and exactly zero initial B."""

    def __init__(self, in_features: int, out_features: int, rank: int, *,
                 scale: float = 1., generator: torch.Generator | None = None,
                 trainable: bool = True):
        super().__init__()
        _positive_int(rank, "rank")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("factor scale must be finite and positive")
        self.rank = rank
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.A = nn.Parameter(torch.empty(rank, in_features), requires_grad=trainable)
        self.B = nn.Parameter(torch.zeros(out_features, rank), requires_grad=trainable)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=generator)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x.to(self.A.dtype), self.A), self.B) * self.scale

    def effective_weight(self) -> Tensor:
        return self.scale * (self.B @ self.A)


class ParallelAdapterLinear(nn.Module):
    """One frozen (possibly bnb) map, parallel learner LTM/STM, frozen roles.

    Explicit role scope is mandatory. No adapter merge or global active-bank
    mutation exists. Frozen role banks may be populated from a bootstrap
    checkpoint before collection; they are never optimizer coordinates.
    """

    def __init__(self, base: nn.Module, *, ltm_rank: int, stm_rank: int,
                 generator: torch.Generator | None = None,
                 ltm_scale: float = 1., stm_scale: float = 1.):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.banks = nn.ModuleDict({
            name: LowRankBank(self.in_features, self.out_features, rank, scale=scale,
                              generator=generator, trainable=trainable)
            for name, rank, scale, trainable in (
                ("ltm", ltm_rank, ltm_scale, True), ("stm", stm_rank, stm_scale, True),
                ("teacher", ltm_rank, ltm_scale, False),
                ("evaluator", ltm_rank, ltm_scale, False))
        })
        # Quantized weight.dtype is storage dtype, not the adapter compute dtype.
        self.banks.to(device=base.weight.device)

    def forward(self, x: Tensor) -> Tensor:
        role = _ROLE.get()
        if role not in ROLES:
            raise RuntimeError("parallel adapters require an explicit model role")
        result = self.base(x)
        names = ("ltm", "stm") if role == "learner" else (role,)
        for name in names:
            result = result + self.banks[name](x).to(result.dtype)
        return result

    def effective_weight(self, role: str) -> Tensor:
        if role not in ROLES:
            raise ValueError("unknown model role")
        if getattr(self.base.weight, "quant_state", None) is not None:
            raise ValueError("dequantize only for diagnostics, not by effective_weight")
        result = self.base.weight
        for name in (("ltm", "stm") if role == "learner" else (role,)):
            result = result + self.banks[name].effective_weight().to(result.dtype)
        return result


class STMLinear(nn.Module):
    """Dense LTM map plus a parallel STM contribution; both remain attached."""

    def __init__(self, in_features: int, out_features: int, *, stm_rank: int,
                 generator: torch.Generator | None = None, stm_scale: float = 1.):
        super().__init__()
        self.ltm = nn.Linear(in_features, out_features)
        self.stm = LowRankBank(in_features, out_features, stm_rank, scale=stm_scale,
                               generator=generator)

    def forward(self, x: Tensor) -> Tensor:
        return self.ltm(x) + self.stm(x).to(x.dtype)

    def effective_weight(self) -> Tensor:
        return self.ltm.weight + self.stm.effective_weight()


class AdaptedLSTMCell(nn.Module):
    """PyTorch i,f,g,o convention, with STM on both declared gate matrices."""

    def __init__(self, input_size: int, hidden_size: int, *, stm_rank: int,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_gates = STMLinear(input_size, 4 * hidden_size, stm_rank=stm_rank,
                                    generator=generator)
        self.hidden_gates = STMLinear(hidden_size, 4 * hidden_size, stm_rank=stm_rank,
                                     generator=generator)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h, c = state
        i, f, g, o = (self.input_gates(x) + self.hidden_gates(h)).chunk(4, dim=-1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c.tanh(), c


@dataclass(frozen=True)
class LanguageEvent:
    text: str
    issued_at: float
    completed_at: float | None = None

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("instruction text must be nonempty")
        if not math.isfinite(self.issued_at) or self.issued_at < 0:
            raise ValueError("instruction issued_at must be finite and nonnegative")
        if self.completed_at is not None and (
                not math.isfinite(self.completed_at) or self.completed_at < self.issued_at):
            raise ValueError("instruction completion must not precede its issue time")


def _events(context: Sequence) -> tuple[LanguageEvent, ...]:
    result = tuple(e if isinstance(e, LanguageEvent) else LanguageEvent(**e) for e in context)
    if any(a.issued_at > b.issued_at for a, b in zip(result, result[1:])):
        raise ValueError("language events must be ordered by issue time")
    return result


def validate_batch(batch: Mapping[str, Any], max_history: int) -> tuple[int, int]:
    images, actions, elapsed = (batch[k] for k in ("images", "executed_actions", "elapsed"))
    if images.ndim != 5 or images.shape[2] != 3 or min(images.shape) < 1:
        raise ValueError("images must have shape [B,T,3,H,W]")
    b, t = images.shape[:2]
    if t > max_history:
        raise ValueError("history exceeds context_length; truncate explicitly and version the recipe")
    if actions.shape != (b, t, 4) or elapsed.shape != (b, t, 1):
        raise ValueError("executed_actions/elapsed must have shapes [B,T,4]/[B,T,1]")
    for x in (images, actions, elapsed):
        if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
            raise ValueError("history tensors must be finite floating point")
    if bool(((images < 0) | (images > 1)).any()) or bool((elapsed < 0).any()):
        raise ValueError("RGB must be in [0,1] and elapsed seconds nonnegative")
    if len(batch["contexts"]) != b:
        raise ValueError("contexts must contain B language-event sequences")
    for context in batch["contexts"]:
        _events(context)
    if "start_time" in batch:
        start = batch["start_time"]
        if start.shape != (b,) or not bool(torch.isfinite(start).all()) or bool((start < 0).any()):
            raise ValueError("start_time must have shape [B] with finite nonnegative seconds")
    return b, t


def render_context(batch: Mapping[str, Any], item: int, step: int, *,
                   max_chars: int = 4096) -> str:
    """Versioned, bounded visible context; never drop an unfinished command.

    Completion is revealed only at/after completed_at. Future events cannot
    affect the prompt. Old completed events can be removed as whole events;
    action history is bounded separately by the explicit raw history window.
    """
    elapsed = batch["elapsed"][item, :step + 1, 0].detach().cpu()
    start = float(batch["start_time"][item]) if "start_time" in batch else 0.
    now = start + float(elapsed.sum())
    active, completed = [], []
    for e in _events(batch["contexts"][item]):
        if e.issued_at > now:
            continue
        done = e.completed_at is not None and e.completed_at <= now
        row = {"text": e.text, "issued_at": e.issued_at,
               "status": "completed" if done else "active"}
        if done:
            row["completed_at"] = e.completed_at
        (completed if done else active).append(row)
    actions = batch["executed_actions"][item, :step + 1].detach().cpu().tolist()
    fixed = {"schema": CONTEXT_SCHEMA_VERSION, "now": now,
             "executed_actions": actions, "elapsed": elapsed.tolist()}

    def serialize(rows):
        return json.dumps({**fixed, "instructions": sorted(rows, key=lambda r: r["issued_at"])},
                          sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    rows = list(active)
    if len(serialize(rows)) > max_chars:
        raise ValueError("context budget cannot retain unfinished instructions and executed history")
    for row in reversed(completed):
        if len(serialize(rows + [row])) <= max_chars:
            rows.append(row)
    return serialize(rows)


def batch_digest(batch: Mapping[str, Any], steps: int | None = None) -> str:
    """Content key, not a caller-supplied observation id or stale feature id."""
    digest = hashlib.sha256()
    t = batch["images"].shape[1] if steps is None else steps
    for key in ("images", "executed_actions", "elapsed", "start_time"):
        if key not in batch:
            continue
        x = batch[key] if key == "start_time" else batch[key][:, :t]
        x = x.detach().cpu().contiguous()
        digest.update(f"{key}:{x.dtype}:{tuple(x.shape)}".encode())
        digest.update(x.view(torch.uint8).numpy().tobytes())
    # Hash only information visible in this prefix, including completion status.
    for b in range(batch["images"].shape[0]):
        for step in range(t):
            digest.update(render_context(batch, b, step).encode())
    return digest.hexdigest()


def _install_adapters(layers, top_layers, ltm_rank, stm_rank, generator):
    if type(top_layers) is not int or not 1 <= top_layers <= len(layers):
        raise ValueError("top_layers must select a nonempty suffix of the text blocks")
    selected = tuple(range(len(layers) - top_layers, len(layers)))
    for index in selected:
        for name in ADAPTER_TARGETS:
            parent_name, leaf = name.rsplit(".", 1)
            parent = layers[index].get_submodule(parent_name)
            base = getattr(parent, leaf)
            if isinstance(base, ParallelAdapterLinear):
                raise ValueError("backbone already has Plan 19 adapters")
            setattr(parent, leaf, ParallelAdapterLinear(base, ltm_rank=ltm_rank,
                    stm_rank=stm_rank, generator=generator))
    return selected


def _float32_top(text, selected_layers):
    """Configure once, without dequantizing frozen 4-bit weight storage.

    Linear4bit returns its INPUT dtype even with FP32 compute, and lazily
    recasts a mismatched bias in forward. Set compute/bias precision up front;
    casting only final features cannot recover small adapter displacements.
    """
    for index in selected_layers:
        layer = text.layers[index]
        layer.input_layernorm.float()
        layer.post_attention_layernorm.float()
        for name in ADAPTER_TARGETS:
            adapter = layer.get_submodule(name)
            adapter.banks.float()
            base = adapter.base
            if hasattr(base, "compute_dtype") and hasattr(base, "compute_type_is_set"):
                base.compute_dtype = torch.float32
                base.compute_type_is_set = True
                if base.bias is not None:
                    base.bias.data = base.bias.data.float()
            else:
                base.float()
    text.norm.float()


class ContextEncoder(nn.Module):
    """Per-role, one-entry inference caches. Training/FD never consult caches.

    No KV caches are used: full multimodal positions are rebuilt per request.
    A cache covers identical raw inputs plus immutable processor/backbone
    provenance. Publications invalidate learner features only. Calls sharing
    the backbone are serialized, including frozen teacher/evaluator calls.
    """

    def __init__(self, *, context_length: int, max_context_chars: int):
        super().__init__()
        _positive_int(context_length, "context_length")
        _positive_int(max_context_chars, "max_context_chars")
        self.context_length = context_length
        self.max_context_chars = max_context_chars
        self.publication = 0
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[str, Tensor, float]] = {}
        self.provenance: dict[str, Any] = {}

    def mark_published(self) -> None:
        with self._lock:
            self.publication += 1
            self._cache.pop("learner", None)

    def clear_caches(self) -> None:
        with self._lock:
            self._cache.clear()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.clear_caches()
        return result

    def inference_features(self, batch, *, role: str) -> tuple[Tensor, dict]:
        if role not in ROLES:
            raise ValueError("unknown model role")
        if torch.is_grad_enabled() or self.training:
            raise RuntimeError("feature cache is inference-only; use eval() and no_grad()")
        validate_batch(batch, self.context_length)
        payload = {"input": batch_digest(batch), "provenance": self.provenance,
                   "context": (CONTEXT_SCHEMA_VERSION, self.max_context_chars),
                   "publication": self.publication if role == "learner" else 0}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self._lock:
            cached = self._cache.get(role)
            hit = cached is not None and cached[0] == key
            if not hit:
                cached = (key, self(batch, role=role).detach(), time.monotonic())
                self._cache[role] = cached
            return cached[1].clone(), {"cache_hit": hit,
                "feature_age_seconds": time.monotonic() - cached[2], "role": role}


class _TinyAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(width, width), nn.Linear(width, width)
        self.v_proj, self.o_proj = nn.Linear(width, width), nn.Linear(width, width)

    def forward(self, x, mask):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        logits = q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return self.o_proj(logits.softmax(-1) @ v)


class _TinyMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(width, 2 * width), nn.Linear(width, 2 * width)
        self.down_proj = nn.Linear(2 * width, width)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attn, self.mlp = _TinyAttention(width), _TinyMLP(width)

    def forward(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class TinyContextEncoder(ContextEncoder):
    """Real trainable CPU smoke encoder, not a Qwen capability substitute."""

    def __init__(self, *, width=16, depth=3, top_layers=1, ltm_rank=4, stm_rank=1,
                 context_length=8, max_context_chars=4096, factor_seed=0):
        super().__init__(context_length=context_length, max_context_chars=max_context_chars)
        self.hidden_size = width
        self.embedding = nn.Embedding(2050, width)
        self.image_projection = nn.Linear(3, width)
        self.action_projection = nn.Linear(5, width)
        self.layers = nn.ModuleList([_TinyBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.requires_grad_(False)
        generator = torch.Generator().manual_seed(factor_seed)
        self.selected_layers = _install_adapters(self.layers, top_layers, ltm_rank,
                                                 stm_rank, generator)
        self.provenance = {"kind": "tiny_cpu", "model_schema": MODEL_SCHEMA_VERSION,
                           "width": width, "depth": depth, "factor_seed": factor_seed}

    def forward(self, batch, *, role="learner") -> Tensor:
        b, t = validate_batch(batch, self.context_length)
        device, dtype = self.embedding.weight.device, self.embedding.weight.dtype
        texts = [render_context(batch, i, j, max_chars=self.max_context_chars)
                 for i in range(b) for j in range(t)]
        # Stable, local word hashing avoids a tokenizer dependency for CPU smoke.
        tokens = [[1 + int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "big") % 2048
                   for w in re.findall(r'\w+|[^\w\s]', text)] for text in texts]
        n = max(map(len, tokens))
        ids = torch.zeros(b * t, n, dtype=torch.long, device=device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(tokens):
            ids[i, :len(row)] = torch.tensor(row, device=device)
            valid[i, :len(row)] = True
        with self._lock, _role_scope(role):
            with torch.no_grad():
                images = batch["images"].reshape(b * t, *batch["images"].shape[2:]).to(device, dtype)
                patches = F.adaptive_avg_pool2d(images, 2).flatten(2).transpose(1, 2)
                visual = self.image_projection(patches)
                control = torch.cat((batch["executed_actions"], batch["elapsed"]), -1)
                control = self.action_projection(control.reshape(b * t, 5).to(device, dtype))[:, None]
                query = self.embedding(torch.full((b * t, 1), 2049, device=device))
                x = torch.cat((visual, self.embedding(ids), control, query), 1)
                keep = torch.cat((torch.ones(b * t, 4, dtype=torch.bool, device=device), valid,
                                  torch.ones(b * t, 2, dtype=torch.bool, device=device)), 1)
                # Padding must not make a record's prediction depend on its peers.
                positions = (keep.long().cumsum(-1) - 1).to(dtype)
                frequencies = torch.arange(1, self.hidden_size + 1, device=device, dtype=dtype)
                x = x + torch.sin(positions[..., None] / frequencies)
                mask = keep[:, None, :] & torch.ones(x.shape[1], x.shape[1], dtype=torch.bool,
                                                    device=device).tril()[None]
                for layer in self.layers[:self.selected_layers[0]]:
                    x = layer(x, mask)
                x = x.detach()
            with nullcontext() if role == "learner" else torch.no_grad():
                for layer in self.layers[self.selected_layers[0]:]:
                    x = layer(x, mask)
                return self.norm(x[:, -1]).reshape(b, t, -1)


class QwenContextEncoder(ContextEncoder):
    """One shared real Qwen2.5-VL with an explicit no-grad lower prefix.

    The split follows installed Transformers 5.2 Qwen2.5-VL source, using its
    own vision, rotary, causal-mask and decoder implementations. No full
    per-layer hidden-state tuple or vocabulary logits are produced for learners.
    """

    def __init__(self, qwen: nn.Module, processor, *, top_layers=1, ltm_rank=4,
                 stm_rank=1, context_length=8, max_context_chars=4096,
                 max_tokens=2048, factor_seed=0):
        super().__init__(context_length=context_length, max_context_chars=max_context_chars)
        self.qwen = qwen.requires_grad_(False).eval()
        self.processor = processor
        self.max_tokens = max_tokens
        text = qwen.model.language_model
        self.hidden_size = text.config.hidden_size
        self.selected_layers = _install_adapters(text.layers, top_layers, ltm_rank,
                         stm_rank, torch.Generator().manual_seed(factor_seed))
        _float32_top(text, self.selected_layers)
        self.compute_policy = {"version": 1, "prefix_dtype": str(text.embed_tokens.weight.dtype),
            "top_dtype": "torch.float32", "top_rotary": "recomputed_float32_at_split",
            "autocast": "disabled", "quantized_weight_storage": "unchanged"}
        self.provenance = {"compute_policy": self.compute_policy}

    def encode_processed(self, inputs: Mapping[str, Tensor], *, role: str) -> Tensor:
        """Stateless full hidden sequence, also used for frozen role scoring."""
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        if role not in ROLES:
            raise ValueError("unknown model role")
        if any(k in inputs for k in ("past_key_values", "inputs_embeds", "pixel_values_videos")):
            raise ValueError("only fresh RGB/text context is supported, never cross-role KV caches")
        core, text = self.qwen.model, self.qwen.model.language_model
        device = text.embed_tokens.weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        ids, attention = inputs["input_ids"], inputs["attention_mask"]
        if ids.ndim != 2 or attention.shape != ids.shape or not bool(attention.bool().any(-1).all()):
            raise ValueError("input_ids and attention_mask must be nonempty [B,S]")
        if ids.shape[1] > self.max_tokens:
            raise ValueError("processed context exceeds max_tokens; no implicit instruction truncation")
        with self._lock, _role_scope(role), torch.autocast(device_type=device.type, enabled=False):
            with torch.no_grad():
                x = core.get_input_embeddings()(ids)
                grid = inputs.get("image_grid_thw")
                if "pixel_values" in inputs:
                    vision = core.get_image_features(inputs["pixel_values"], grid, return_dict=True)
                    image = torch.cat(vision.pooler_output, 0).to(x.device, x.dtype)
                    image_mask, _ = core.get_placeholder_mask(ids, x, image_features=image)
                    x = x.masked_scatter(image_mask, image)
                positions, _ = core.get_rope_index(ids, image_grid_thw=grid,
                                                  attention_mask=attention)
                cache_position = torch.arange(ids.shape[1], device=device)
                mask_kwargs = dict(config=text.config, inputs_embeds=x, attention_mask=attention,
                                   cache_position=cache_position, past_key_values=None, position_ids=None)
                masks = {"full_attention": create_causal_mask(**mask_kwargs)}
                if text.has_sliding_layers:
                    masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
                rotary = text.rotary_emb(x, positions)
                layer_kwargs = dict(position_embeddings=rotary, position_ids=None,
                                    past_key_values=None, use_cache=False,
                                    cache_position=cache_position, output_attentions=False)
                for layer in text.layers[:self.selected_layers[0]]:
                    x = layer(x, attention_mask=masks[layer.attention_type], **layer_kwargs)[0]
                x = x.detach().float()
                # Recompute, rather than upcast rounded BF16 trigonometric values.
                layer_kwargs["position_embeddings"] = text.rotary_emb(x, positions)
                mask_kwargs["inputs_embeds"] = x
                masks = {"full_attention": create_causal_mask(**mask_kwargs)}
                if text.has_sliding_layers:
                    masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
            with nullcontext() if role == "learner" else torch.no_grad():
                for layer in text.layers[self.selected_layers[0]:]:
                    x = layer(x, attention_mask=masks[layer.attention_type], **layer_kwargs)[0]
                return text.norm(x)

    def prepare(self, batch, *, suffix: str = "") -> tuple[dict[str, Tensor], tuple[int, int]]:
        b, t = validate_batch(batch, self.context_length)
        if self.processor is None:
            raise RuntimeError("raw multimodal inference requires the checkpoint's pinned processor")
        messages, images = [], []
        for i in range(b):
            for j in range(t):
                context = render_context(batch, i, j, max_chars=self.max_context_chars)
                conversation = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": context + suffix}]}]
                messages.append(self.processor.apply_chat_template(conversation, tokenize=False,
                                                                  add_generation_prompt=True))
                images.append(batch["images"][i, j].detach().cpu())
        processed = dict(self.processor(text=messages, images=images, padding=True,
                                       return_tensors="pt", do_rescale=False))
        return processed, (b, t)

    def forward(self, batch, *, role="learner") -> Tensor:
        inputs, (b, t) = self.prepare(batch)
        hidden = self.encode_processed(inputs, role=role)
        mask = inputs["attention_mask"].to(hidden.device)
        last = (torch.arange(mask.shape[1], device=hidden.device)[None] * mask).max(-1).values
        return hidden[torch.arange(b * t, device=hidden.device), last].reshape(b, t, -1)


@dataclass(frozen=True)
class RecurrentState:
    h: Tensor
    c: Tensor
    publication: int
    history_digest: str
    steps: int
    owner_id: int


class StaleStateError(ValueError):
    pass


class PiCarStudent(nn.Module):
    """Shared encoder, dense LTM recurrence/readouts, additive STM matrices.

    forward is pure with respect to module state and supports functional_call
    on every trainable coordinate. predict is a separate inference-only API;
    stale streaming states fail, and predict(batch) reconstructs from zero.
    """

    def __init__(self, encoder: ContextEncoder, *, hidden_size=64, return_bins=21,
                 stm_rank=1, factor_seed=0):
        super().__init__()
        _positive_int(hidden_size, "hidden_size")
        _positive_int(return_bins, "return_bins")
        self.encoder = encoder
        self.hidden_size, self.return_bins = hidden_size, return_bins
        self.distillation_receipt: dict[str, Any] | None = None
        generator = torch.Generator().manual_seed(factor_seed)
        self.projection = STMLinear(encoder.hidden_size + 5, hidden_size,
                                    stm_rank=stm_rank, generator=generator)
        self.recurrent = AdaptedLSTMCell(hidden_size, hidden_size, stm_rank=stm_rank,
                                         generator=generator)
        self.return_head = STMLinear(hidden_size, return_bins, stm_rank=stm_rank,
                                     generator=generator)
        self.action_head = STMLinear(hidden_size, 7, stm_rank=stm_rank, generator=generator)

    def train(self, mode: bool = True):
        super().train(mode)
        # Trainable adapters stay differentiable in eval mode; disable all
        # pretrained stochastic layers for replay/FD endpoint pairing.
        self.encoder.eval()
        return self

    def _run(self, batch, features, state=None):
        b, t = features.shape[:2]
        reference = self.projection.ltm.weight
        features = features.to(reference.device, reference.dtype)
        control = torch.cat((batch["executed_actions"], batch["elapsed"]), -1).to(features)
        x = torch.tanh(self.projection(torch.cat((features, control), -1)))
        h, c = (features.new_zeros(b, self.hidden_size), features.new_zeros(b, self.hidden_size)) \
            if state is None else (state.h.to(features), state.c.to(features))
        for j in range(0 if state is None else t - 1, t):
            h, c = self.recurrent(x[:, j], (h, c))
        return {"return_logits": self.return_head(h), "action_mean": self.action_head(h)}, (h, c)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        with torch.autocast(device_type=self.projection.ltm.weight.device.type, enabled=False):
            return self._run(batch, self.encoder(batch, role="learner"))[0]

    @torch.no_grad()
    def predict(self, batch: Mapping[str, Any], state: RecurrentState | None = None):
        if self.training:
            raise RuntimeError("predict requires eval mode")
        _, t = validate_batch(batch, self.encoder.context_length)
        if state is not None and (state.owner_id != id(self)
                or state.publication != self.encoder.publication
                or state.steps != t - 1 or state.history_digest != batch_digest(batch, t - 1)):
            raise StaleStateError("stale recurrent state; reconstruct with predict(batch) without state")
        if state is not None and (state.h.shape != (batch["images"].shape[0], self.hidden_size)
                or state.c.shape != state.h.shape or not bool(torch.isfinite(state.h).all())
                or not bool(torch.isfinite(state.c).all())):
            raise StaleStateError("invalid recurrent state; reconstruct from raw history")
        features, metadata = self.encoder.inference_features(batch, role="learner")
        with torch.autocast(device_type=self.projection.ltm.weight.device.type, enabled=False):
            output, (h, c) = self._run(batch, features, state)
        new_state = RecurrentState(h.detach().clone(), c.detach().clone(), self.encoder.publication,
                                   batch_digest(batch), t, id(self))
        return output, new_state, metadata

    def mark_published(self) -> None:
        """Caller must serialize optimizer publication with all role inference."""
        self.encoder.mark_published()

    def get_extra_state(self):
        return {"model_schema": MODEL_SCHEMA_VERSION,
                "distillation_receipt": self.distillation_receipt,
                "publication": self.encoder.publication}

    def set_extra_state(self, state):
        if state["model_schema"] != MODEL_SCHEMA_VERSION:
            raise ValueError("incompatible model checkpoint schema")
        self.distillation_receipt = state["distillation_receipt"]
        self.encoder.publication = state["publication"]
        self.encoder.clear_caches()

    def parameter_banks(self) -> dict[str, tuple[str, ...]]:
        banks: dict[str, list[str]] = {"ltm": [], "stm": []}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                bank = "stm" if ".stm." in name else "ltm"
                banks[bank].append(name)
        return {key: tuple(value) for key, value in banks.items()}

    def trainable_inventory(self) -> list[dict[str, Any]]:
        banks = self.parameter_banks()
        return [{"name": name, "shape": list(p.shape), "numel": p.numel(),
                 "dtype": str(p.dtype), "bank": "stm" if name in banks["stm"] else "ltm"}
                for name, p in self.named_parameters() if p.requires_grad]


def build_tiny_model(seed: int = 0, *, hidden_size=16, return_bins=21, top_layers=1,
                     ltm_rank=4, stm_rank=1, context_length=8, width=16, depth=3,
                     dtype=torch.float32, device="cpu") -> PiCarStudent:
    """Deterministic CPU model, independent named dense and factor RNGs."""
    seeds = {name: derive_component_seed(seed, "picar." + name) for name in
             ("backbone_initialization", "student_initialization", "transformer_factors", "recurrent_factors")}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seeds["backbone_initialization"])
        encoder = TinyContextEncoder(width=width, depth=depth, top_layers=top_layers,
            ltm_rank=ltm_rank, stm_rank=stm_rank, context_length=context_length,
            factor_seed=seeds["transformer_factors"])
        torch.random.default_generator.manual_seed(seeds["student_initialization"])
        student = PiCarStudent(encoder, hidden_size=hidden_size, return_bins=return_bins,
                              stm_rank=stm_rank, factor_seed=seeds["recurrent_factors"])
    encoder.provenance["seeds"] = seeds
    return student.to(device=device, dtype=dtype)


def checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    """Verify local source bytes, including weights, processor and tokenizer."""
    root = Path(path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        weights = ["model.safetensors"]
    names = weights + [p.name for p in sorted(root.iterdir())
                       if p.suffix in (".json", ".txt", ".jinja", ".model", ".tiktoken")]
    hashes = {}
    for name in sorted(set(names)):
        target = root / name
        if Path(name).name != name or not target.is_file():
            raise ValueError(f"missing or unsafe checkpoint file: {name}")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = {"sha256": digest.hexdigest(), "bytes": target.stat().st_size}
    return {"files": hashes, "sha256": hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def load_qwen_model(model_path: str | Path = QWEN_MODEL_ID, *, revision: str | None = QWEN_REVISION,
                    local_files_only=False, cache_dir=QWEN_CACHE_DIR, quantized=True,
                    device="cuda:0", compute_dtype=torch.bfloat16, seed=0,
                    top_layers=1, ltm_rank=4, stm_rank=1, hidden_size=64,
                    context_length=8, return_bins=21, max_tokens=2048,
                    min_pixels=56 * 56, max_pixels=224 * 224) -> PiCarStudent:
    """Explicit actual loader; no fallback to tiny models or unquantized Qwen.

    A local directory is identified by verified content hashes, not by an
    unverifiable Hub ref. Repository loading requires an exact 40-hex revision.
    The explicit loader downloads a pinned snapshot into this project's cache
    by default. local_files_only=True provides strict offline operation.
    PEFT is unnecessary: parallel banks are implemented directly over frozen
    bnb maps, avoiding mutable PEFT active-adapter state and recursive LoRA.
    """
    source = str(model_path)
    local = Path(source).is_dir()
    if isinstance(model_path, Path) and not local:
        raise FileNotFoundError(f"explicit model directory does not exist: {model_path}")
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be an exact 40-hex model commit")
    if not local and revision is None:
        raise ValueError("repository loading requires a pinned exact model revision")
    if compute_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unsupported Qwen compute dtype")
    _positive_int(top_layers, "top_layers")
    # Optional dependencies are intentionally imported only at this entry point.
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from huggingface_hub import snapshot_download

    resolved = source if local else snapshot_download(repo_id=source, revision=revision,
        local_files_only=local_files_only, cache_dir=str(cache_dir))
    if not local and Path(resolved).name != revision:
        raise RuntimeError("resolved checkpoint does not match the pinned model revision")
    common = dict(local_files_only=True,
                  cache_dir=str(cache_dir), trust_remote_code=False)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=["lm_head", "visual"]) if quantized else None
    manifest = checkpoint_manifest(resolved)
    processor = AutoProcessor.from_pretrained(resolved, min_pixels=min_pixels,
                                               max_pixels=max_pixels, **common)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(resolved,
        quantization_config=quantization, device_map={"": str(device)}, dtype=compute_dtype,
        attn_implementation="eager", **common)
    if quantized and not getattr(qwen, "is_loaded_in_4bit", False):
        raise RuntimeError("requested quantization was not applied")
    seeds = {name: derive_component_seed(seed, "picar." + name) for name in
             ("student_initialization", "transformer_factors", "recurrent_factors")}
    with torch.random.fork_rng(devices=[]):
        encoder = QwenContextEncoder(qwen, processor, top_layers=top_layers, ltm_rank=ltm_rank,
            stm_rank=stm_rank, context_length=context_length, max_tokens=max_tokens,
            factor_seed=seeds["transformer_factors"])
        torch.random.default_generator.manual_seed(seeds["student_initialization"])
        student = PiCarStudent(encoder, hidden_size=hidden_size, return_bins=return_bins,
                               stm_rank=stm_rank, factor_seed=seeds["recurrent_factors"])
    # Keep all trainable coordinates float32, including those over quantized maps.
    for module in (student.projection, student.recurrent, student.return_head, student.action_head):
        module.to(device=device, dtype=torch.float32)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers")}
    if quantized:
        versions["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
    encoder.provenance = {"kind": "qwen2_5_vl", "source": source,
        "declared_revision": revision, "revision_verified": not local,
        "resolved_path": str(resolved), "cache_dir": str(cache_dir),
        "checkpoint_manifest": manifest, "local_files_only": local_files_only,
        "processor_source": source, "tokenizer_source": source,
        "processor_pixels": [min_pixels, max_pixels], "max_tokens": max_tokens,
        "quantization": quantization.to_dict() if quantization else None,
        "software": versions, "seeds": seeds, "selected_layers": list(encoder.selected_layers),
        "adapter_targets": list(ADAPTER_TARGETS), "factor_scales": {"ltm": 1., "stm": 1.},
        "initialization": "random_A_zero_B", "model_schema": MODEL_SCHEMA_VERSION,
        "compute_policy": encoder.compute_policy,
        "trainable_inventory": student.trainable_inventory(), "kv_cache": "disabled"}
    return student
