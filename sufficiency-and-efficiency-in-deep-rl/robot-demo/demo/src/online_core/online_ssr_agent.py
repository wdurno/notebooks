import math

import torch

try:
    from src.core.lanczos import l_lanczos
    from src.core.ssr_agent import SSRAgent
except ModuleNotFoundError:
    from core.lanczos import l_lanczos
    from core.ssr_agent import SSRAgent


class OnlineSSRAgent(SSRAgent):
    """EMA-based online SSR variant.

    Concrete subclasses still define `loss`, optimizer construction, and any
    model-specific target-network cadence. This class only owns the online
    sufficient-statistics mechanics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_grad_vector = None
        self.ssr_weight_sq_sum = None
        self.ssr_effective_n = None
        self.ssr_last_pi = None

    def ssr_dict(self):
        d = super().ssr_dict()
        d.update(
            {
                "online_ssr_mean_scaled": True,
                "ssr_weight_sq_sum": self.ssr_weight_sq_sum,
                "ssr_effective_n": self.ssr_effective_n,
                "ssr_last_pi": self.ssr_last_pi,
            }
        )
        return d

    def load_ssr_dict(self, d):
        super().load_ssr_dict(d)
        if d.get("online_ssr_mean_scaled", False):
            self.ssr_weight_sq_sum = d.get("ssr_weight_sq_sum")
            self.ssr_effective_n = d.get("ssr_effective_n")
            self.ssr_last_pi = d.get("ssr_last_pi")
            self.current_grad_vector = None
            return
        if self.ssr_n is not None and self.ssr_n > 0:
            scale = float(self.ssr_n)
            if self.ssr_low_rank_matrix is not None:
                self.ssr_low_rank_matrix = self.ssr_low_rank_matrix / math.sqrt(scale)
            if self.ssr_residual_diagonal is not None:
                self.ssr_residual_diagonal = self.ssr_residual_diagonal / scale
            self.ssr_weight_sq_sum = 1.0 / scale
            self.ssr_effective_n = float(scale)
        else:
            self.ssr_weight_sq_sum = None
            self.ssr_effective_n = None
        self.ssr_last_pi = None
        self.current_grad_vector = None

    def _get_get_grad_generator(self, n=None, random_idx=False):
        del n, random_idx

        def get_grad_generator():
            def grad_generator():
                yield self._require_current_grad_vector()

            return grad_generator

        return get_grad_generator

    def _require_current_grad_vector(self):
        if self.current_grad_vector is None:
            raise RuntimeError("OnlineSSRAgent requires a cached gradient before memorize().")
        return self.current_grad_vector

    def _cache_current_grad_vector(self):
        grads = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            if grad is None:
                grad = torch.zeros_like(parameter, device=parameter.device)
            grads.append(grad.reshape([-1, 1]))
        if not grads:
            raise RuntimeError("OnlineSSRAgent found no trainable parameters while caching gradients.")
        self.current_grad_vector = torch.cat(grads, dim=0).clone().detach()
        return self.current_grad_vector

    def _after_optimizer_step(self):
        return None

    def _update_center_statistics(self):
        self.ssr_prev_center = self.ssr_center.to(self.gpu_saver) if self.ssr_center is not None else None
        self.ssr_center = self.get_param().clone().detach()
        if self.ssr_model_dimension is None:
            self.ssr_model_dimension = self.ssr_center.shape[0]
        if self.ssr_prev_center is None:
            return None
        rescale = (self.dt_mean_N - 1) / self.dt_mean_N
        if isinstance(self.dt_mean_trend, torch.Tensor) and self.dt_mean_trend.shape == torch.Size([]):
            self.dt_mean_trend = float(self.dt_mean_trend)
        prev_center = self.ssr_prev_center.to(self.ssr_center.device)
        delta = self.ssr_center - prev_center
        self.dt_mean_trend *= rescale
        self.dt_mean_trend += delta / self.dt_mean_N
        self.dt_mean_norm_trend *= rescale
        self.dt_mean_norm_trend += delta.pow(2).sum() / self.dt_mean_N
        self.dt_mean_trace_cov *= rescale
        self.dt_mean_trace_cov += (delta - self.dt_mean_trend).pow(2).sum() / self.dt_mean_N
        self.ssr_cov_trace = (delta * delta).sum() if self.ssr_cov_trace is None else self.ssr_cov_trace + (delta * delta).sum()
        self.ssr_cov_n = 1 if self.ssr_cov_n is None else self.ssr_cov_n + 1
        return None

    def memorize(self, pi=None, disable_tqdm=False):
        del disable_tqdm
        gradient = self._require_current_grad_vector().to(self.device)
        self._update_center_statistics()
        if self.ssr_model_dimension is None:
            self.ssr_model_dimension = gradient.shape[0]
        if pi is None:
            pi_value = float(self.optimal_pi())
        else:
            pi_value = float(pi)
        pi_value = max(0.0, min(1.0, pi_value))
        gradient = gradient.reshape([-1, 1])
        if self.ssr_low_rank_matrix is None or self.ssr_residual_diagonal is None:
            self.ssr_low_rank_matrix, self.ssr_residual_diagonal = l_lanczos(
                get_grad_generator=None,
                r=min(self.ssr_rank, self.ssr_model_dimension),
                p=self.ssr_model_dimension,
                device=self.device,
                mfi_alternate=lambda x: gradient.matmul(gradient.transpose(0, 1).matmul(x)),
                diag_alternate=lambda: gradient * gradient,
                calc_diag=True,
            )
            self.ssr_weight_sq_sum = 1.0
            self.ssr_effective_n = 1.0
        else:
            prev_low_rank = self.ssr_low_rank_matrix
            prev_residual = self.ssr_residual_diagonal.to(self.device)

            def mfi_ema(x):
                prev_term = prev_low_rank.matmul(prev_low_rank.transpose(0, 1).matmul(x))
                prev_term += prev_residual * x
                new_term = gradient.matmul(gradient.transpose(0, 1).matmul(x))
                return (1.0 - pi_value) * prev_term + pi_value * new_term

            def diag_ema():
                prev_diag = (prev_low_rank * prev_low_rank).sum(dim=1, keepdim=True) + prev_residual
                return (1.0 - pi_value) * prev_diag + pi_value * (gradient * gradient)

            self.ssr_low_rank_matrix, self.ssr_residual_diagonal = l_lanczos(
                get_grad_generator=None,
                r=min(self.ssr_rank, self.ssr_model_dimension),
                p=self.ssr_model_dimension,
                device=self.device,
                mfi_alternate=mfi_ema,
                diag_alternate=diag_ema,
                calc_diag=True,
            )
            weight_sq_sum = 1.0 if self.ssr_weight_sq_sum is None else float(self.ssr_weight_sq_sum)
            self.ssr_weight_sq_sum = (1.0 - pi_value) ** 2 * weight_sq_sum + pi_value ** 2
            self.ssr_effective_n = 1.0 / self.ssr_weight_sq_sum if self.ssr_weight_sq_sum > 0.0 else None
        self.ssr_last_pi = pi_value
        self.ssr_n = 1 if self.ssr_n is None else self.ssr_n + 1
        return None

    def ssr(self, lmbda=None):
        del lmbda
        if self.ssr_low_rank_matrix is None or self.ssr_residual_diagonal is None:
            return 0.0
        d = self.get_param() - self.ssr_center
        low_rank = self.ssr_low_rank_matrix
        residual = self.ssr_residual_diagonal
        d_t_a = d.transpose(0, 1).matmul(low_rank)
        a_t_d = d_t_a.transpose(0, 1)
        d_t_res_d = (d * residual).transpose(0, 1).matmul(d)
        return 0.5 * (d_t_a.matmul(a_t_d) + d_t_res_d)

    def _trace_fisher_inverse(self, diagonal_eps=1e-8):
        if self.ssr_residual_diagonal is None:
            return None
        residual = torch.clamp(self.ssr_residual_diagonal.reshape([-1, 1]).to(self.device), min=diagonal_eps)
        trace_d_inv = (1.0 / residual).sum()
        if self.ssr_low_rank_matrix is None:
            return trace_d_inv
        low_rank = self.ssr_low_rank_matrix.to(self.device)
        if low_rank.numel() == 0:
            return trace_d_inv
        d_inv_a = low_rank / residual
        gram = torch.eye(low_rank.shape[1], device=low_rank.device, dtype=low_rank.dtype)
        gram = gram + low_rank.transpose(0, 1).matmul(d_inv_a)
        d_inv_sq_a = low_rank / (residual * residual)
        correction_matrix = low_rank.transpose(0, 1).matmul(d_inv_sq_a)
        correction = torch.trace(torch.linalg.solve(gram, correction_matrix))
        return torch.clamp(trace_d_inv - correction, min=0.0)

    def optimal_pi(self, pi_min=0.0, pi_max=1.0, eps=1e-6):
        if self.ssr_effective_n is None or self.dt_mean_norm_trend == 0.0:
            pi = torch.tensor(0.5, device=self.device)
        else:
            trace_inverse = self._trace_fisher_inverse()
            if trace_inverse is None:
                pi = torch.tensor(0.5, device=self.device)
            else:
                numerator = trace_inverse / max(float(self.ssr_effective_n), eps)
                denominator = 2.0 * self.dt_mean_norm_trend + eps
                pi = 1.0 - numerator / denominator
                pi = torch.as_tensor(pi, device=self.device).clone().detach()
        if float(pi) < pi_min:
            pi = torch.tensor(pi_min, device=self.device)
        if float(pi) > pi_max:
            pi = torch.tensor(pi_max, device=self.device)
        return pi

    def fit(self, data, iters=1, pi=None, memorize=True, grad_clip=None):
        if iters < 0:
            raise ValueError("iters must be non-negative")
        self.train()
        pi_value = float(pi) if pi is not None else float(self.optimal_pi())
        self.ssr_last_pi = pi_value
        loss = None
        if iters == 0:
            if memorize:
                self.memorize(pi=pi_value)
            return float(pi_value), 0.0
        self.optimizer.zero_grad()
        for _ in range(iters):
            loss = self.loss(data) / max(iters, 1)
            loss = pi_value * loss + (1.0 - pi_value) * self.ssr() / max(iters, 1)
            loss.backward()
        self._cache_current_grad_vector()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.parameters() if parameter.requires_grad],
                max_norm=float(grad_clip),
            )
        self.optimizer.step()
        self._after_optimizer_step()
        if memorize:
            self.memorize(pi=pi_value)
        return float(pi_value), float(loss.detach())
