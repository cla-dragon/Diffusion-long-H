from typing import Optional, Union, Callable

import numpy as np
import torch

from cleandiffuser.diffusion.diffusionsde import (
    ContinuousDiffusionSDE,
    SUPPORTED_SOLVERS,
    SUPPORTED_SAMPLING_STEP_SCHEDULE,
    xtheta_to_epstheta,
    epstheta_to_xtheta,
)


class CurvatureContinuousDiffusionSDE(ContinuousDiffusionSDE):
    """Continuous diffusion SDE with curvature logging for OOD ranking.

    Curvature is approximated as the sum of step-wise epsilon variation norms:
        sum_i ||epsilon_i - epsilon_{i+1}||_2
    where epsilon_i is the model's noise prediction at denoising step i.
    """

    def sample(
            self,
            prior: torch.Tensor,
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
            use_ema: bool = True,
            temperature: float = 1.0,
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            diffusion_x_sampling_steps: int = 0,
            warm_start_reference: Optional[torch.Tensor] = None,
            warm_start_forward_level: float = 0.3,
            requires_grad: bool = False,
            preserve_history: bool = False,
            return_curvature_per_step: bool = False,
            curvature_normalize_by_dt: bool = False,
            **kwargs,
    ):
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None,
        }

        model = self.model if not use_ema else self.model_ema

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            warm_start_forward_level = self.epsilon + warm_start_forward_level * (1. - self.epsilon)
            fwd_alpha, fwd_sigma = self.noise_schedule_funcs["forward"](
                torch.ones((1,), device=self.device) * warm_start_forward_level,
                **(self.noise_schedule_params or {}),
            )
            xt = warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            xt = torch.randn_like(prior) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.detach().cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            t_diffusion = [self.t_diffusion[0], warm_start_forward_level]
        else:
            t_diffusion = self.t_diffusion

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps
                )
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(t_diffusion, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {})
        )
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        curvature = torch.zeros((n_samples,), dtype=prior.dtype, device=self.device)
        curvature_steps = []
        prev_eps_theta = None
        prev_t_scalar = None
        prev_i = None

        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):
            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

            pred, logp = self.guided_sampling(
                xt,
                t,
                alphas[i],
                sigmas[i],
                model,
                condition_vec_cfg,
                w_cfg,
                condition_vec_cg,
                w_cg,
                requires_grad,
            )

            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            if prev_eps_theta is not None and prev_i != i:
                eps_delta = (prev_eps_theta - eps_theta).reshape(n_samples, -1)
                step_curvature = torch.linalg.vector_norm(eps_delta, ord=2, dim=1)
                if curvature_normalize_by_dt and prev_t_scalar is not None:
                    dt = (prev_t_scalar - sample_step_schedule[i]).abs().clamp_min(1e-8)
                    step_curvature = step_curvature / dt
                curvature = curvature + step_curvature
                if return_curvature_per_step:
                    curvature_steps.append(step_curvature.detach().cpu())
            prev_eps_theta = eps_theta.detach()
            prev_t_scalar = sample_step_schedule[i]
            prev_i = i

            if solver == "ddpm":
                xt = (
                    (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta)
                    + (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta
                )
                if i > 1:
                    xt += stds[i] * torch.randn_like(xt)

            elif solver == "ddim":
                xt = alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta

            elif solver == "ode_dpmsolver_1":
                xt = (alphas[i - 1] / alphas[i]) * xt - sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver_1":
                xt = (
                    (alphas[i - 1] / alphas[i]) * xt
                    - 2 * sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta
                    + sigmas[i - 1] * torch.expm1(2 * hs[i]).sqrt() * torch.randn_like(xt)
                )

            elif solver == "sde_dpmsolver++_1":
                xt = (
                    (sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt
                    - alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta
                    + sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt)
                )

            elif solver == "sde_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (
                        (sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt
                        - alphas[i - 1] * torch.expm1(-2 * hs[i]) * D
                        + sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt)
                    )
                else:
                    xt = (
                        (sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt
                        - alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta
                        + sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt)
                    )

            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.detach().cpu().numpy()

        if self.classifier is not None and w_cg != 0.:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        log["curvature"] = curvature.detach().cpu().numpy()
        if return_curvature_per_step:
            if len(curvature_steps) > 0:
                log["curvature_per_step"] = torch.stack(curvature_steps, dim=1).numpy()
            else:
                log["curvature_per_step"] = np.zeros((n_samples, 0), dtype=np.float32)

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
