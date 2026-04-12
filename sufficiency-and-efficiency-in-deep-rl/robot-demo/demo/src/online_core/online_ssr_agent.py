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
        self.ssr_inv_trace = None
        self.ssr_inv_trace_weight_sq_sum = None

    def ssr_dict(self):
        d = super().ssr_dict()
        d.update(
            {
                "online_ssr_mean_scaled": True,
                "ssr_weight_sq_sum": self.ssr_weight_sq_sum,
                "ssr_effective_n": self.ssr_effective_n,
                "ssr_last_pi": self.ssr_last_pi,
                "ssr_inv_trace": self.ssr_inv_trace,
                "ssr_inv_trace_weight_sq_sum": self.ssr_inv_trace_weight_sq_sum,
            }
        )
        return d

    def load_ssr_dict(self, d):
        super().load_ssr_dict(d)
        if d.get("online_ssr_mean_scaled", False):
            self.ssr_weight_sq_sum = d.get("ssr_weight_sq_sum")
            self.ssr_effective_n = d.get("ssr_effective_n")
            self.ssr_last_pi = d.get("ssr_last_pi")
            self.ssr_inv_trace = d.get("ssr_inv_trace")
            self.ssr_inv_trace_weight_sq_sum = d.get("ssr_inv_trace_weight_sq_sum")
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
        self.ssr_inv_trace = None
        self.ssr_inv_trace_weight_sq_sum = None
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
        prev_center = self.ssr_prev_center.to(self.ssr_center.device)
        delta = self.ssr_center - prev_center
        pi_prev = 1.0 if self.ssr_last_pi is None else float(self.ssr_last_pi)
        pi_prev = max(0.0, min(1.0, pi_prev))
        mu_prev = self.dt_mean_trend
        if not isinstance(mu_prev, torch.Tensor) or mu_prev.shape == torch.Size([]):
            mu_prev = torch.zeros_like(delta)
        else:
            mu_prev = mu_prev.to(delta.device)
        residual = delta - mu_prev
        self.dt_mean_trend = (1.0 - pi_prev) * mu_prev + pi_prev * delta
        self.dt_mean_norm_trend = (1.0 - pi_prev) * float(self.dt_mean_norm_trend) + pi_prev * float(delta.pow(2).sum())
        self.dt_mean_trace_cov = (1.0 - pi_prev) * float(self.dt_mean_trace_cov) + pi_prev * float(residual.pow(2).sum())
        if self.ssr_inv_trace_weight_sq_sum is None or self.ssr_last_pi is None:
            self.ssr_inv_trace_weight_sq_sum = 1.0
        else:
            self.ssr_inv_trace_weight_sq_sum = (1.0 - pi_prev) ** 2 * float(self.ssr_inv_trace_weight_sq_sum) + pi_prev ** 2
        if pi_prev > 0.0 and self.ssr_inv_trace_weight_sq_sum > 0.0:
            self.ssr_inv_trace = self.dt_mean_trace_cov / (pi_prev * self.ssr_inv_trace_weight_sq_sum)
        self.ssr_cov_trace = (delta * delta).sum() if self.ssr_cov_trace is None else self.ssr_cov_trace + (delta * delta).sum()
        self.ssr_cov_n = 1 if self.ssr_cov_n is None else self.ssr_cov_n + 1
        return None

    def memorize(self, pi=None, disable_tqdm=False):
        del disable_tqdm # kept only for SSRAgent interface compatibility; unused in the online EMA formulation 
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
        del lmbda  # kept only for SSRAgent interface compatibility; unused in the online EMA formulation
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
        """Return the current local estimate of ``tr(I(theta)^{-1})``.

        Usage:
            This helper is consumed by ``optimal_pi()``. It should be called
            only after the online agent has observed at least one parameter
            increment, because the estimate is built from the history of
            parameter-center updates gathered in ``_update_center_statistics()``.

        Implicit inputs:
            - ``self.ssr_center`` and ``self.ssr_prev_center`` for the observed
              parameter increment sequence,
            - ``self.dt_mean_trend`` for the local EMA drift estimate,
            - ``self.dt_mean_trace_cov`` for the EMA estimate of centered
              increment norm-squared,
            - ``self.ssr_last_pi`` for the previous online gain, defaulting to
              ``1`` on the first usable update, and
            - ``self.ssr_inv_trace_weight_sq_sum`` for the squared EMA weight
              mass needed to convert a weighted residual variance into an
              effective-sample-size scaled trace estimate.

        Output:
            A nonnegative scalar estimate of ``tr(I(theta)^{-1})`` as a tensor
            on ``self.device``, or ``None`` if insufficient online state has
            been accumulated yet.

        Side effects:
            None. All state mutation happens upstream in
            ``_update_center_statistics()``. This method only reads the current
            online statistics and packages them as a tensor result.

        Mathematical justification:
            ``AGENTS.md`` already develops the local Gaussianized single-
            observation update law and the EMA locality argument. Reusing that
            same local theory, if

            ``Delta_k = theta_k - theta_{k-1}``

            is centered by a local EMA drift estimate ``mu_k``, then the
            residual increment satisfies

            ``Cov(Delta_k - mu_k) approx (pi_{k-1} / n_eff,k) I(theta_k)^{-1}``.

            Taking traces gives

            ``E ||Delta_k - mu_k||^2 approx (pi_{k-1} / n_eff,k) tr(I(theta_k)^{-1})``.

            Therefore we do not estimate a full inverse Fisher matrix here.
            Instead, we keep an EMA estimate of the centered increment squared
            norm together with the corresponding squared EMA weight mass, then
            recover the local inverse-Fisher trace through the associated
            effective sample size. This avoids the numerical instability of
            inverting a low-rank-plus-diagonal Fisher approximation with extreme
            eigenvalue skew.
        """
        del diagonal_eps
        if self.ssr_inv_trace is None:
            return None
        return torch.as_tensor(self.ssr_inv_trace, device=self.device).clone().detach()

    def optimal_pi(self, pi_min=0.0, pi_max=1.0, eps=1e-6):
        """Estimate the locally optimal online forgetting/control weight.

        This method implements the scalar rule described in
        ``src/online_core/AGENTS.md``:

        ``pi* = 1 - tr(I(theta)^{-1}) / (n_eff * (2 ||d theta||^2 + eps))``

        using online diagnostics already tracked by ``OnlineSSRAgent``.

        Inputs:
            - ``pi_min`` / ``pi_max``: clipping bounds for the returned value.
            - ``eps``: small numerical stabilizer used only in the denominator
              of the closed-form rule.

        Output:
            A scalar tensor on ``self.device`` representing the clipped online
            estimate of ``pi``.

        Side effects:
            None. This method only reads current sufficient-statistic state.

        Notes:
            If the user supplies ``pi`` directly to ``fit()``, this method need
            not be called. When insufficient online state exists, a conservative
            default of ``0.5`` is returned.
        """
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

    def fit(self, loss, pi=None, memorize=True, grad_clip=None):
        """Apply one already-computed online loss and optional EMA memorization.

        Args:
            loss: Scalar pre-backward task loss tensor. This should be the
                concrete agent's already-computed objective for the current
                observation/transition before SSR regularization is mixed in.
                The tensor must still be attached to the current computation
                graph, and `backward()` must not have been called on it yet.
            pi: Optional user override for the online forgetting/control weight.
                If omitted, `optimal_pi()` is used.
            memorize: If True, update the EMA Fisher estimate after the optimizer
                step using the raw unclipped gradient cached from this backward
                pass.
            grad_clip: Optional gradient clipping threshold applied after the raw
                gradient is cached and before `optimizer.step()`.

        Returns:
            `(pi_value, loss_value)` where `pi_value` is the scalar weight used
            for this step and `loss_value` is the detached scalar value of the
            SSR-regularized loss actually backpropagated.

        Notes:
            Environment interaction, transition construction, and task-loss
            assembly remain the responsibility of the concrete implementation.
            This method owns only the standardized post-loss mechanics.
        """
        self.train()
        if loss.ndim != 0:
            raise ValueError("OnlineSSRAgent.fit expects a scalar loss tensor with loss.ndim == 0.")
        pi_value = float(pi) if pi is not None else float(self.optimal_pi())
        self.ssr_last_pi = pi_value
        self.optimizer.zero_grad()
        loss = pi_value * loss + (1.0 - pi_value) * self.ssr()
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
