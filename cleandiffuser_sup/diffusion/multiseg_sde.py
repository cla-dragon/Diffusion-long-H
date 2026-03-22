from typing import Optional

import torch

from cleandiffuser.diffusion.diffusionsde import at_least_ndim
from .repaint_sde import RepaintContinuousDiffusionSDE


class MultiSegmentRepaintDiffusionSDE(RepaintContinuousDiffusionSDE):
    """RePaint diffusion with training-time mixed boundary conditioning and
    inference-time multi-segment overlap stitching.
    """

    def _side_noise(self, x_side: torch.Tensor, t_side: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.noise_schedule_funcs["forward"](t_side, **(self.noise_schedule_params or {}))
        alpha = at_least_ndim(alpha, x_side.dim())
        sigma = at_least_ndim(sigma, x_side.dim())
        return alpha * x_side + sigma * torch.randn_like(x_side)

    def _build_overlap_weights(
        self,
        overlap_length: int,
        blend_type: str,
        exp_beta: float,
        device,
        dtype,
    ) -> torch.Tensor:
        if blend_type in ["avg", "mean"]:
            return torch.full((1, overlap_length, 1), 0.5, device=device, dtype=dtype)

        if overlap_length == 1:
            return torch.ones((1, 1, 1), device=device, dtype=dtype)

        t = torch.linspace(0.0, 1.0, overlap_length, device=device, dtype=dtype)
        if blend_type in ["exponential", "exp"]:
            beta = max(float(exp_beta), 1e-6)
            tail = torch.exp(torch.tensor(-beta, device=device, dtype=dtype))
            w = (torch.exp(-beta * t) - tail) / (1.0 - tail)
        elif blend_type == "cosine":
            w = 0.5 * (1.0 + torch.cos(torch.pi * t))
        elif blend_type == "linear":
            w = 1.0 - t
        elif blend_type == "smoothstep":
            w = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
        else:
            raise ValueError(f"Unsupported overlap blend_type: {blend_type}")

        return w.view(1, overlap_length, 1)

    def loss(
        self,
        x0,
        condition=None,
        overlap_length: int = 4,
        overlap_prob: float = 0.5,
        inpaint_start_prob: float = 0.5,
        inpaint_end_prob: float = 0.5,
        **kwargs,
    ):
        xt, t, eps = self.add_noise(x0)
        bsz, horizon, _ = x0.shape
        overlap_length = int(max(1, min(overlap_length, horizon - 1)))

        device = x0.device
        rand_overlap_left = torch.rand((bsz,), device=device) < overlap_prob
        rand_overlap_right = torch.rand((bsz,), device=device) < overlap_prob

        rand_inpaint_left = (~rand_overlap_left) & (torch.rand((bsz,), device=device) < inpaint_start_prob)
        rand_inpaint_right = (~rand_overlap_right) & (torch.rand((bsz,), device=device) < inpaint_end_prob)

        # Overlap mode: inject a noisy boundary chunk as a hard condition during training.
        if rand_overlap_left.any():
            t_left = t[rand_overlap_left]
            left_chunk = x0[rand_overlap_left, :overlap_length, :]
            xt[rand_overlap_left, :overlap_length, :] = self._side_noise(left_chunk, t_left)

        if rand_overlap_right.any():
            t_right = t[rand_overlap_right]
            right_chunk = x0[rand_overlap_right, -overlap_length:, :]
            xt[rand_overlap_right, -overlap_length:, :] = self._side_noise(right_chunk, t_right)

        # Inpaint mode: pin endpoint tokens so the model sees strict endpoint constraints.
        if rand_inpaint_left.any():
            xt[rand_inpaint_left, 0, :] = x0[rand_inpaint_left, 0, :]

        if rand_inpaint_right.any():
            xt[rand_inpaint_right, -1, :] = x0[rand_inpaint_right, -1, :]

        condition_vec = self.model["condition"](condition) if condition is not None else None

        if self.predict_noise:
            pred = self.model["diffusion"](xt, t, condition_vec)
            loss = (pred - eps) ** 2
        else:
            pred = self.model["diffusion"](xt, t, condition_vec)
            loss = (pred - x0) ** 2

        loss = loss * self.loss_weight * (1 - self.fix_mask)

        # No denoising loss on strict inpaint anchors.
        if rand_inpaint_left.any() or rand_inpaint_right.any():
            anchor_mask = torch.ones_like(loss)
            if rand_inpaint_left.any():
                anchor_mask[rand_inpaint_left, 0, :] = 0.0
            if rand_inpaint_right.any():
                anchor_mask[rand_inpaint_right, -1, :] = 0.0
            loss = loss * anchor_mask

        weighted_regression_tensor = kwargs.get("weighted_regression_tensor", None)
        if weighted_regression_tensor is not None:
            loss *= weighted_regression_tensor.unsqueeze(-1)

        return loss.mean()

    @torch.no_grad()
    def sample_multi_segment(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        num_segments: int,
        segment_horizon: int,
        overlap_length: int,
        inner_resample_rounds: int = 4,
        overlap_blend_type: str = "avg",
        overlap_exp_beta: float = 3.0,
        repaint_times: int = 5,
        jump_len: int = 5,
        solver: str = "ddpm",
        sample_steps: int = 40,
        use_ema: bool = True,
        temperature: float = 1.0,
        condition_cg=None,
        w_cg: float = 0.0,
    ):
        if num_segments < 2:
            raise ValueError("num_segments must be >= 2.")

        n_samples, obs_dim = start_state.shape
        overlap_length = int(max(1, min(overlap_length, segment_horizon - 1)))
        blend_w = self._build_overlap_weights(
            overlap_length=overlap_length,
            blend_type=overlap_blend_type,
            exp_beta=overlap_exp_beta,
            device=self.device,
            dtype=start_state.dtype,
        )

        segments = [
            torch.randn((n_samples, segment_horizon, obs_dim), device=self.device, dtype=start_state.dtype)
            for _ in range(num_segments)
        ]
        candidate_overlap_cost = torch.zeros((n_samples,), device=self.device, dtype=start_state.dtype)

        for _ in range(max(1, inner_resample_rounds)):
            prev_segments = [s.clone() for s in segments]
            updated_segments = []

            for idx in range(num_segments):
                prior = torch.zeros((n_samples, segment_horizon, obs_dim), device=self.device, dtype=start_state.dtype)
                mask = torch.zeros_like(prior)

                if idx == 0:
                    prior[:, 0, :] = start_state
                    mask[:, 0, :] = 1.0

                if idx == num_segments - 1:
                    prior[:, -1, :] = goal_state
                    mask[:, -1, :] = 1.0

                if idx > 0:
                    left_overlap = prev_segments[idx - 1][:, -overlap_length:, :]
                    prior[:, :overlap_length, :] = left_overlap
                    mask[:, :overlap_length, :] = 1.0

                if idx < num_segments - 1:
                    right_overlap = prev_segments[idx + 1][:, :overlap_length, :]
                    prior[:, -overlap_length:, :] = right_overlap
                    mask[:, -overlap_length:, :] = 1.0

                x_seg, _ = self.sample(
                    prior=prior,
                    mask=mask,
                    repaint_times=repaint_times,
                    jump_len=jump_len,
                    solver=solver,
                    n_samples=n_samples,
                    sample_steps=sample_steps,
                    use_ema=use_ema,
                    temperature=temperature,
                    condition_cg=condition_cg,
                    w_cg=w_cg,
                )
                updated_segments.append(x_seg)

            # Compute overlap disagreement per candidate before enforcing fusion.
            round_cost = torch.zeros((n_samples,), device=self.device, dtype=start_state.dtype)
            for idx in range(1, num_segments):
                left_overlap = updated_segments[idx - 1][:, -overlap_length:, :]
                right_overlap = updated_segments[idx][:, :overlap_length, :]
                round_cost += ((left_overlap - right_overlap) ** 2).mean(dim=(1, 2))
            candidate_overlap_cost += round_cost

            # Enforce overlap fusion after each inner round.
            for idx in range(1, num_segments):
                left_overlap = updated_segments[idx - 1][:, -overlap_length:, :]
                right_overlap = updated_segments[idx][:, :overlap_length, :]
                fused_overlap = blend_w * left_overlap + (1.0 - blend_w) * right_overlap
                updated_segments[idx - 1][:, -overlap_length:, :] = fused_overlap
                updated_segments[idx][:, :overlap_length, :] = fused_overlap

            segments = updated_segments

        stitched = [segments[0]]
        for idx in range(1, num_segments):
            stitched.append(segments[idx][:, overlap_length:, :])
        full_traj = torch.cat(stitched, dim=1)

        candidate_sorted_indices = torch.argsort(candidate_overlap_cost, dim=0)
        return full_traj, {
            "segments": segments,
            "candidate_overlap_cost": candidate_overlap_cost,
            "candidate_sorted_indices": candidate_sorted_indices,
        }
