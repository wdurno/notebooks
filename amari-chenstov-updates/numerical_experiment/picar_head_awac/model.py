"""Small heads and exact memoization of a frozen causal representation."""

from collections import OrderedDict
import hashlib

import torch
from torch import nn
from torch.func import functional_call

from numerical_experiment.picar.models import STMLinear, batch_digest


class Heads(nn.Module):
    def __init__(self, hidden=64, width=12, seed=199001):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed + 1)
            self.actor = STMLinear(hidden, 7, stm_rank=1, generator=generator)
            self.q1 = nn.Sequential(STMLinear(hidden + 7, width, stm_rank=1, generator=generator),
                                    nn.Tanh(), STMLinear(width, 1, stm_rank=1, generator=generator))
            self.q2 = nn.Sequential(STMLinear(hidden + 7, width, stm_rank=1, generator=generator),
                                    nn.Tanh(), STMLinear(width, 1, stm_rank=1, generator=generator))

    def forward(self, features, actions):
        q_input = torch.cat((features, actions), -1)
        return {"mean": self.actor(features), "q1": self.q1(q_input).squeeze(-1),
                "q2": self.q2(q_input).squeeze(-1)}


class Coordinates:
    def __init__(self, heads):
        self.heads = heads
        self.names = tuple(dict(heads.named_parameters()))
        self.shapes = {n: p.shape for n, p in heads.named_parameters()}
        self.sizes = {n: p.numel() for n, p in heads.named_parameters()}
        indices, offset = {"ltm": [], "stm": []}, 0
        for name in self.names:
            size = self.sizes[name]
            indices["stm" if ".stm." in name else "ltm"].extend(range(offset, offset + size))
            offset += size
        self.indices = {k: torch.tensor(v, dtype=torch.long) for k, v in indices.items()}

    def vector(self):
        return torch.cat([p.detach().flatten() for p in self.heads.parameters()])

    def parameters(self, vector):
        values, offset = {}, 0
        for name in self.names:
            size = self.sizes[name]
            values[name] = vector[offset:offset + size].reshape(self.shapes[name])
            offset += size
        return values

    def predict(self, vector, features, actions):
        return functional_call(self.heads, self.parameters(vector), (features, actions))

    def publish(self, vector):
        with torch.no_grad():
            for name, parameter in self.heads.named_parameters():
                parameter.copy_(self.parameters(vector)[name])

    def inventory(self):
        return [{"name": n, "shape": list(self.shapes[n]), "count": self.sizes[n],
                 "bank": "stm" if ".stm." in n else "ltm"} for n in self.names]


class FrozenFeatures:
    def __init__(self, model, identity, capacity=1024):
        self.model = model.eval().requires_grad_(False)
        self.identity = str(identity)
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    @torch.no_grad()
    def __call__(self, history):
        key = hashlib.sha256((self.identity + batch_digest(history)).encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key].clone()
        self.misses += 1
        features, _ = self.model.encoder.inference_features(history, role="learner")
        _, (hidden, _) = self.model._run(history, features)
        result = hidden[0].detach().cpu().clone()
        self.cache[key] = result
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result.clone()

    def clear(self):
        self.cache.clear()
