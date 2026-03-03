from typing import Optional, Union, Callable
import torch
import numpy as np
from cleandiffuser.diffusion.diffusionsde import ContinuousDiffusionSDE, SUPPORTED_SOLVERS, SUPPORTED_SAMPLING_STEP_SCHEDULE, xtheta_to_epstheta, epstheta_to_xtheta

class RepaintContinuousDiffusionSDE(ContinuousDiffusionSDE):

    def get_repaint_schedule(self, sample_steps, jump_len, repaint_times):
        """
        Generate the sequence of time steps for RePaint.
        Returns a list of tuples: (current_step_idx, next_step_idx)
        """
        schedule = []
        t = sample_steps
        
        while t > 0:
            # Determine the jump size (cannot go below 0)
            # steps are 1-based indices in alphas array. 0 is not used for sampling start.
            curr_jump = min(jump_len, t) # We want to reach t - curr_jump
            
            # The exact number of steps to denoise depends on where we stop.
            # We want to go from t down to t - curr_jump.
            # If t - curr_jump == 0, we finish.
            
            # Repaint Loop
            for u in range(repaint_times):
                # 1. Denoise phase: t -> t-1 -> ... -> t - curr_jump
                for k in range(curr_jump):
                    curr = t - k
                    next = t - k - 1
                    # If next is 0, it means we reached x0 (or final step)
                    schedule.append((curr, next))
                
                # 2. Noise phase (Backtrack): t - curr_jump -> ... -> t
                # Only if we are not in the last repetition AND not finishing the generation
                if u < repaint_times - 1 and (t - curr_jump) > 0:
                    for k in range(curr_jump):
                        curr = t - curr_jump + k
                        next = t - curr_jump + k + 1
                        schedule.append((curr, next))
            
            # Advance the main pointer
            t -= curr_jump
            
        return schedule

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # ---------- Repaint Mask (NEW) ---------- #
            mask: Optional[torch.Tensor] = None,
            # ---------- Repaint Times (NEW) ---------- #
            repaint_times: int = 1,
            # ---------- Jump Length (NEW) ---------- #
            jump_len: int = 1,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ----------- Diffusion-X sampling ----------
            diffusion_x_sampling_steps: int = 0,
            # ----------- Warm-Starting -----------
            warm_start_reference: Optional[torch.Tensor] = None,
            warm_start_forward_level: float = 0.3,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        """Sampling with Repaint support.
        
        Args:
            mask: Tensor of shape (1, *x_shape) or (n_samples, *x_shape). 
                  1 indicates the pixel is known (from prior), 0 indicates unknown (to be generated).
            repaint_times: (U) Number of resampling steps. Recommended: 5~10.
            jump_len: (j) Jump length for resampling. Recommended: 10 (if sample_steps is large enough).
        """
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."
        
        # Check compatibility for RePaint
        if repaint_times > 1 or jump_len > 1:
            if "ode" in solver or "2M" in solver:
                print(f"[Warning] You are using RePaint (U={repaint_times}, j={jump_len}) with solver '{solver}'. "
                      f"High-order ODE solvers are NOT compatible. Please use 'ddpm', 'ddim' or 'sde_dpmsolver_1'.")

        # ===================== Initialization =====================
        prior = prior.to(self.device)
        # Handle mask
        if mask is not None:
            mask = mask.to(self.device)
            # Ensure mask is broadcastable
            if mask.ndim < prior.ndim:
                mask = mask.unsqueeze(0)
        
        # If mask is None but fix_mask is present, we wrap it
        if mask is None and self.fix_mask is not None:
             # This is a bit tricky because fix_mask might be 0.
             pass

        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        # Warm start logic
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            warm_start_forward_level = self.epsilon + warm_start_forward_level * (1. - self.epsilon)
            fwd_alpha, fwd_sigma = self.noise_schedule_funcs["forward"](
                torch.ones((1,), device=self.device) * warm_start_forward_level, **(self.noise_schedule_params or {}))
            xt = warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            xt = torch.randn_like(prior) * temperature
        
        # Initial mixing for step T (if mask is present)
        # Just to be safe, we can mix at the very beginning if it's pure noise, 
        # but usually the loop handles it.
        # However, if we start from random noise, the known part is lost. 
        # We should init known part to noisy prior.
        if mask is not None:
            # Init known part
            # At step T (which is index sample_steps in our schedule logic, but index 0 in history)
            # We assume alpha_T approx 0.
            pass

        # If using base class fix_mask logic (when mask is None):
        if mask is None:
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            t_diffusion = [self.t_diffusion[0], warm_start_forward_level]
        else:
            t_diffusion = self.t_diffusion
        
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(t_diffusion, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================
        # Generate the RePaint schedule
        # It contains tuples (curr_idx, next_idx)
        # curr_idx > next_idx: Denoise
        # curr_idx < next_idx: Diffuse (Noise)
        
        # Note: Standard Diffusion-X logic is usually prepended. We keep it simple here.
        schedule = self.get_repaint_schedule(sample_steps, jump_len, repaint_times)
        
        for (i, next_i) in schedule:
            
            # ---------------- Case 1: Denoise (Reverse) ----------------
            if i > next_i:
                t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

                # guided sampling
                pred, logp = self.guided_sampling(
                    xt, t, alphas[i], sigmas[i],
                    model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad)

                # clip the prediction
                pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

                # transform to eps_theta
                eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
                x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

                # one-step update: xt (at i) -> xt_prev (at next_i)
                # NOTE: The solver logic in base class assumes step i -> i-1. 
                # Here next_i is exactly i-1 in our schedule construction for Denoise phase.
                # So we can reuse standard solvers. 
                # If next_i != i-1 (e.g. multistep jump), standard solver fails. 
                # Our schedule guarantees i -> i-1 for Denoise.
                
                if solver == "ddpm":
                    xt = (
                            (alphas[next_i] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                            (sigmas[next_i] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                    if next_i > 0: # Add noise if not final step (0)
                        xt += (stds[i] * torch.randn_like(xt))

                elif solver == "ddim":
                    xt = (alphas[next_i] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[next_i] * eps_theta)

                elif solver == "ode_dpmsolver_1":
                    xt = (alphas[next_i] / alphas[i]) * xt - sigmas[next_i] * torch.expm1(hs[i]) * eps_theta
                
                elif solver == "sde_dpmsolver_1":
                    xt = ((alphas[next_i] / alphas[i]) * xt -
                        2 * sigmas[next_i] * torch.expm1(hs[i]) * eps_theta +
                        sigmas[next_i] * torch.expm1(2 * hs[i]).sqrt() * torch.randn_like(xt))
                
                # We skip higher order solvers or complicated ones for RePaint flexibility
                else: 
                     # Fallback to DDPM/SDE-1 for other names as they are most robust for jumping
                     # Or just execute DDIM logic
                     xt = (alphas[next_i] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[next_i] * eps_theta)

                # Apply Mask Mixing (Crucial Step)
                if mask is not None:
                    # Sample noise for the known part at level next_i
                    if next_i > 0:
                        noise = torch.randn_like(prior)
                        xt_known = alphas[next_i] * prior + sigmas[next_i] * noise
                    else:
                        xt_known = prior
                    
                    xt = mask * xt_known + (1. - mask) * xt
                
                # Base class fix mask
                elif mask is None and self.fix_mask is not None:
                    xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

            # ---------------- Case 2: Diffuse (Forward / Backtrack) ----------------
            elif i < next_i:
                # We need to go from known xt (at i) to unknown xt_next (at next_i)
                # This corresponds to q(x_{next}|x_{curr})
                # Using the formula derived from variance schedule:
                # x_{next} = (alpha_{next}/alpha_{i}) * x_{i} + sigma_{rel} * noise
                
                coef = alphas[next_i] / alphas[i]
                # sigma_{ next | i }^2 = sigma_{next}^2 - coef^2 * sigma_{i}^2
                sigma_rel = (sigmas[next_i]**2 - coef**2 * sigmas[i]**2).clamp(min=0).sqrt()
                
                xt = coef * xt + sigma_rel * torch.randn_like(xt)
                
                # No mask mixing needed here strictly, but you could mix if you want strict adherence to known x_{next}.
                # But x_{next} (the higher noise level) will be fixed in the subsequent Denoise step anyway.
            
            # ---------------- Logging ----------------
            if preserve_history:
                # Map scheduling step to history index? 
                # It's tricky because schedule is longer than sample_steps.
                # We just overwrite or ignore. Here we ignore intermediate jumps for clean history,
                # or we just log if i matches a step.
                if i - 1 < log["sample_history"].shape[1]:
                     log["sample_history"][:, sample_steps - i] = xt.detach().cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None and w_cg != 0.:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
