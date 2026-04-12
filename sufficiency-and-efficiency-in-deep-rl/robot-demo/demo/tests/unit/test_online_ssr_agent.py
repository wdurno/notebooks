from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.lanczos import l_lanczos
from core.replay_buffer import ReplayBuffer
from core.ssr_agent import SSRAgent
from online_core.online_ssr_agent import OnlineSSRAgent


class SimpleBatch:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y


class ToyOnlineAgent(OnlineSSRAgent):
    def __init__(self):
        super().__init__(replay_buffer=ReplayBuffer(capacity=4), ssr_rank=1, gpu_saver=True, dt_mean_N=4)
        self.linear = torch.nn.Linear(1, 1, bias=False)
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        self.post_step_calls = 0

    def loss(self, batch: SimpleBatch):
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        return ((self.linear(x) - y) ** 2).mean()

    def _after_optimizer_step(self):
        self.post_step_calls += 1
        return None


class OverrideGeneratorAgent(SSRAgent):
    def __init__(self, replay_buffer: ReplayBuffer):
        super().__init__(replay_buffer=replay_buffer, ssr_rank=1, gpu_saver=True, dt_mean_N=4)
        self.linear = torch.nn.Linear(1, 1, bias=False)
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        self.override_called = False

    def loss(self, transitions):
        x = transitions.x.to(self.device).float().reshape([-1, 1])
        y = transitions.y.to(self.device).float().reshape([-1, 1])
        return ((self.linear(x) - y) ** 2).mean()

    def _get_get_grad_generator(self, n=None, random_idx=False):
        del n, random_idx
        self.override_called = True

        def get_grad_generator():
            def grad_generator():
                yield torch.ones_like(self.get_param())

            return grad_generator

        return get_grad_generator


def test_l_lanczos_uses_diag_alternate_without_replaying_gradients():
    def mfi_alternate(x):
        diagonal = torch.tensor([[4.0], [1.0]])
        return diagonal * x

    def diag_alternate():
        return torch.tensor([[4.0], [1.0]])

    low_rank, residual = l_lanczos(
        get_grad_generator=lambda: (_ for _ in ()).throw(RuntimeError("should not use gradients")),
        r=1,
        p=2,
        mfi_alternate=mfi_alternate,
        diag_alternate=diag_alternate,
        calc_diag=True,
    )

    assert low_rank.shape == (2, 1)
    assert residual.shape == (2, 1)
    assert torch.all(residual >= 0.0)


def test_ssr_agent_memoize_uses_overrideable_get_grad_generator():
    replay_buffer = ReplayBuffer(capacity=4)
    replay_buffer.add(torch.tensor([[1.0]]), torch.tensor([1]))
    agent = OverrideGeneratorAgent(replay_buffer)

    agent.memorize(n=1, random_idx=False, disable_tqdm=True)

    assert agent.override_called
    assert agent.ssr_low_rank_matrix is not None
    assert agent.ssr_residual_diagonal is not None


def test_online_memorize_updates_ema_statistics():
    agent = ToyOnlineAgent()
    batch1 = SimpleBatch(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    batch2 = SimpleBatch(torch.tensor([[2.0]]), torch.tensor([[0.0]]))

    agent.fit(batch1, iters=1, pi=0.25, memorize=True)
    assert agent.ssr_n == 1
    assert agent.ssr_weight_sq_sum == pytest.approx(1.0)
    assert agent.ssr_effective_n == pytest.approx(1.0)
    assert agent.ssr_last_pi == pytest.approx(0.25)
    assert agent.current_grad_vector is not None

    agent.fit(batch2, iters=1, pi=0.25, memorize=True)
    assert agent.ssr_n == 2
    assert agent.ssr_weight_sq_sum == pytest.approx(0.625)
    assert agent.ssr_effective_n == pytest.approx(1.6)
    assert agent.post_step_calls == 2


def test_online_optimal_pi_uses_effective_sample_size():
    agent = ToyOnlineAgent()
    agent.ssr_residual_diagonal = torch.tensor([[2.0]], device=agent.device)
    agent.ssr_low_rank_matrix = None
    agent.ssr_effective_n = 4.0
    agent.dt_mean_norm_trend = 2.0

    pi = agent.optimal_pi()

    assert float(pi) == pytest.approx(0.96875, rel=1e-5)


def test_online_fit_respects_user_supplied_pi_without_calling_optimal_pi(monkeypatch):
    agent = ToyOnlineAgent()
    batch = SimpleBatch(torch.tensor([[1.0]]), torch.tensor([[0.0]]))

    def fail_optimal_pi(*args, **kwargs):
        raise AssertionError("optimal_pi should not be called when pi is supplied")

    monkeypatch.setattr(agent, "optimal_pi", fail_optimal_pi)

    pi_value, loss_value = agent.fit(batch, iters=1, pi=0.3, memorize=True)

    assert pi_value == pytest.approx(0.3)
    assert loss_value >= 0.0
