import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from cleandiffuser.diffusion.diffusionsde import (
    DiscreteDiffusionSDE,
    SUPPORTED_SAMPLING_STEP_SCHEDULE,
    epstheta_to_xtheta,
    xtheta_to_epstheta,
)
from cleandiffuser.utils import at_least_ndim


class SegmentStitchDiscreteDiffusionSDE(DiscreteDiffusionSDE):
    """Discrete diffusion with CDGS-style segment composition."""

    def __init__(
        self,
        *args,
        overlap_steps: int,
        condition_guidance_scale: float = 2.0,
        train_overlap_prob: float = 0.5,
        train_side_drop_prob: float = 0.15,
        gsc_inner_loops: int = 10,
        gsc_keep_ratio: float = 0.1,
        gsc_filter_start: float = 0.94,
        gsc_inversion_ratio: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.overlap_steps = int(overlap_steps)
        self.condition_guidance_scale = float(condition_guidance_scale)
        self.train_overlap_prob = float(train_overlap_prob)
        self.train_side_drop_prob = float(train_side_drop_prob)
        self.gsc_inner_loops = int(gsc_inner_loops)
        self.gsc_keep_ratio = float(gsc_keep_ratio)
        self.gsc_filter_start = float(gsc_filter_start)
        self.gsc_inversion_ratio = float(gsc_inversion_ratio)
        self.latest_gsc_scores = None

    @property
    def segment_horizon(self) -> int:
        return int(self.fix_mask.shape[1]) if isinstance(self.fix_mask, torch.Tensor) else int(self.loss_weight.shape[1])

    def _normalize_step(self, step: Optional[torch.Tensor]) -> torch.Tensor:
        if step is None:
            return torch.zeros((1,), device=self.device, dtype=torch.float32)
        step = step.to(device=self.device, dtype=torch.float32)
        denom = max(1, self.diffusion_steps - 1)
        return step / float(denom)

    def _empty_chunk(self, batch_size: int, obs_dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.overlap_steps, obs_dim), device=device, dtype=dtype)

    def _state_code(self, mode: str, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        codes = {
            "drop": torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device),
            "overlap": torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device),
            "anchor": torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device),
        }
        if mode not in codes:
            raise ValueError(f"Unsupported boundary mode: {mode}")
        return codes[mode].repeat(batch_size, 1)

    def _make_condition_payload(
        self,
        *,
        batch_size: int,
        obs_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        left_mode: str,
        right_mode: str,
        left_chunk: Optional[torch.Tensor] = None,
        right_chunk: Optional[torch.Tensor] = None,
        left_step: Optional[torch.Tensor] = None,
        right_step: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        payload = {
            "left_chunk": self._empty_chunk(batch_size, obs_dim, dtype, device),
            "right_chunk": self._empty_chunk(batch_size, obs_dim, dtype, device),
            "left_level": torch.zeros((batch_size,), device=device, dtype=torch.float32),
            "right_level": torch.zeros((batch_size,), device=device, dtype=torch.float32),
            "left_state": self._state_code(left_mode, batch_size, dtype, device),
            "right_state": self._state_code(right_mode, batch_size, dtype, device),
        }
        if left_chunk is not None:
            payload["left_chunk"] = left_chunk.to(device=device, dtype=dtype)
        if right_chunk is not None:
            payload["right_chunk"] = right_chunk.to(device=device, dtype=dtype)
        if left_step is not None:
            payload["left_level"] = self._normalize_step(left_step).to(device=device, dtype=torch.float32)
        if right_step is not None:
            payload["right_level"] = self._normalize_step(right_step).to(device=device, dtype=torch.float32)
        return payload

    def _drop_overlap_payload(self, payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stripped = {k: v.clone() for k, v in payload.items()}
        stripped["left_chunk"].zero_()
        stripped["right_chunk"].zero_()
        stripped["left_level"].zero_()
        stripped["right_level"].zero_()

        left_overlap = payload["left_state"][:, 1] > 0.5
        right_overlap = payload["right_state"][:, 1] > 0.5
        if left_overlap.any():
            stripped["left_state"][left_overlap] = self._state_code(
                "drop",
                int(left_overlap.sum().item()),
                payload["left_state"].dtype,
                payload["left_state"].device,
            )
        if right_overlap.any():
            stripped["right_state"][right_overlap] = self._state_code(
                "drop",
                int(right_overlap.sum().item()),
                payload["right_state"].dtype,
                payload["right_state"].device,
            )
        return stripped

    def _has_overlap_payload(self, payload: Optional[Dict[str, torch.Tensor]]) -> bool:
        if payload is None:
            return False
        return bool((payload["left_state"][:, 1] > 0.5).any() or (payload["right_state"][:, 1] > 0.5).any())

    def _extract_overlap_pair(self, traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return traj[:, : self.overlap_steps, :], traj[:, -self.overlap_steps :, :]

    def _apply_anchor(
        self,
        traj: torch.Tensor,
        start_state: Optional[torch.Tensor] = None,
        goal_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        traj = traj.clone()
        if start_state is not None:
            traj[:, 0, :] = start_state
        if goal_state is not None:
            traj[:, -1, :] = goal_state
        return traj

    def _predict_with_context(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        model,
        payload: Optional[Dict[str, torch.Tensor]],
        condition_cg=None,
        w_cg: float = 0.0,
    ) -> torch.Tensor:
        if payload is None:
            pred = model["diffusion"](xt, t, None)
        elif self._has_overlap_payload(payload):
            cond_vec = model["condition"](payload)
            drop_vec = model["condition"](self._drop_overlap_payload(payload))
            dual_x = torch.cat([xt, xt], dim=0)
            dual_t = torch.cat([t, t], dim=0)
            dual_cond = torch.cat([cond_vec, drop_vec], dim=0)
            dual_pred = model["diffusion"](dual_x, dual_t, dual_cond)
            pred_cond, pred_drop = dual_pred[: xt.shape[0]], dual_pred[xt.shape[0] :]
            pred = pred_drop + self.condition_guidance_scale * (pred_cond - pred_drop)
        else:
            pred = model["diffusion"](xt, t, model["condition"](payload))

        pred, _ = self.classifier_guidance(
            xt,
            t,
            alpha,
            sigma,
            model,
            condition_cg,
            w_cg,
            pred,
        )
        return self.clip_prediction(pred, xt, alpha, sigma)

    def _single_reverse_step(
        self,
        xt: torch.Tensor,
        t_idx: int,
        prev_idx: int,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
        alpha_prev: torch.Tensor,
        sigma_prev: torch.Tensor,
        std_t: torch.Tensor,
        solver: str,
        model,
        payload: Optional[Dict[str, torch.Tensor]],
        condition_cg=None,
        w_cg: float = 0.0,
    ) -> torch.Tensor:
        t = torch.full((xt.shape[0],), t_idx, dtype=torch.long, device=xt.device)
        pred = self._predict_with_context(xt, t, alpha_t, sigma_t, model, payload, condition_cg, w_cg)
        eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alpha_t, sigma_t, pred)

        if solver == "ddpm":
            xt_prev = (
                (alpha_prev / alpha_t) * (xt - sigma_t * eps_theta)
                + (sigma_prev.square() - std_t.square() + 1e-8).sqrt() * eps_theta
            )
            if prev_idx > 0:
                xt_prev = xt_prev + std_t * torch.randn_like(xt_prev)
            return xt_prev

        if solver == "ddim":
            return alpha_prev * ((xt - sigma_t * eps_theta) / alpha_t) + sigma_prev * eps_theta

        raise ValueError(f"Unsupported solver for segment stitching: {solver}")

    def _forward_jump(
        self,
        xt: torch.Tensor,
        alpha_from: torch.Tensor,
        sigma_from: torch.Tensor,
        alpha_to: torch.Tensor,
        sigma_to: torch.Tensor,
    ) -> torch.Tensor:
        coef = alpha_to / alpha_from
        sigma_rel = (sigma_to.square() - coef.square() * sigma_from.square()).clamp_min(0.0).sqrt()
        return coef * xt + sigma_rel * torch.randn_like(xt)

    def _resolve_schedule(
        self,
        sample_steps: int,
        sample_step_schedule: Union[str, Callable] = "uniform",
    ) -> torch.Tensor:
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule not in SUPPORTED_SAMPLING_STEP_SCHEDULE:
                raise ValueError(f"Unsupported sampling schedule: {sample_step_schedule}")
            schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](self.diffusion_steps, sample_steps)
        elif callable(sample_step_schedule):
            schedule = sample_step_schedule(self.diffusion_steps, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")
        return schedule.to(device=self.device, dtype=torch.long)

    def _sample_boundary_noise(self, clean_chunk: torch.Tensor, step_idx: torch.Tensor) -> torch.Tensor:
        step_idx = step_idx.to(device=clean_chunk.device, dtype=torch.long)
        alpha = at_least_ndim(self.alpha[step_idx], clean_chunk.dim())
        sigma = at_least_ndim(self.sigma[step_idx], clean_chunk.dim())
        return alpha * clean_chunk + sigma * torch.randn_like(clean_chunk)

    def _expand_step(self, step_idx: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if step_idx.ndim == 0:
            step_idx = step_idx.repeat(batch_size)
        return step_idx.to(device=device, dtype=torch.long)

    def loss(self, x0, condition=None, **kwargs):
        xt, t, eps = self.add_noise(x0)
        batch_size, horizon, obs_dim = x0.shape
        if self.overlap_steps >= horizon:
            raise ValueError("overlap_steps must be smaller than the segment horizon.")

        left_drop = torch.rand((batch_size,), device=x0.device) < self.train_side_drop_prob
        right_drop = torch.rand((batch_size,), device=x0.device) < self.train_side_drop_prob

        left_overlap = (torch.rand((batch_size,), device=x0.device) < self.train_overlap_prob) & (~left_drop)
        right_overlap = (torch.rand((batch_size,), device=x0.device) < self.train_overlap_prob) & (~right_drop)
        left_anchor = (~left_overlap) & (~left_drop)
        right_anchor = (~right_overlap) & (~right_drop)

        left_step = torch.clamp(t - torch.randint(0, 2, (batch_size,), device=x0.device), min=0)
        right_step = torch.clamp(t - torch.randint(0, 2, (batch_size,), device=x0.device), min=0)

        left_chunk = self._empty_chunk(batch_size, obs_dim, x0.dtype, x0.device)
        right_chunk = self._empty_chunk(batch_size, obs_dim, x0.dtype, x0.device)
        if left_overlap.any():
            noisy_left = self._sample_boundary_noise(x0[left_overlap, : self.overlap_steps, :], left_step[left_overlap])
            left_chunk[left_overlap] = noisy_left
        if right_overlap.any():
            noisy_right = self._sample_boundary_noise(x0[right_overlap, -self.overlap_steps :, :], right_step[right_overlap])
            right_chunk[right_overlap] = noisy_right

        xt = xt.clone()
        if left_anchor.any():
            xt[left_anchor, 0, :] = x0[left_anchor, 0, :]
        if right_anchor.any():
            xt[right_anchor, -1, :] = x0[right_anchor, -1, :]

        payload = self._make_condition_payload(
            batch_size=batch_size,
            obs_dim=obs_dim,
            dtype=x0.dtype,
            device=x0.device,
            left_mode="drop",
            right_mode="drop",
        )
        payload["left_state"][left_overlap] = self._state_code("overlap", int(left_overlap.sum().item()), x0.dtype, x0.device)
        payload["right_state"][right_overlap] = self._state_code("overlap", int(right_overlap.sum().item()), x0.dtype, x0.device)
        payload["left_state"][left_anchor] = self._state_code("anchor", int(left_anchor.sum().item()), x0.dtype, x0.device)
        payload["right_state"][right_anchor] = self._state_code("anchor", int(right_anchor.sum().item()), x0.dtype, x0.device)
        payload["left_chunk"] = left_chunk
        payload["right_chunk"] = right_chunk
        if left_overlap.any():
            payload["left_level"][left_overlap] = self._normalize_step(left_step[left_overlap]).to(
                device=x0.device,
                dtype=torch.float32,
            )
        if right_overlap.any():
            payload["right_level"][right_overlap] = self._normalize_step(right_step[right_overlap]).to(
                device=x0.device,
                dtype=torch.float32,
            )

        pred = self.model["diffusion"](xt, t, self.model["condition"](payload))
        loss = (pred - eps) ** 2 if self.predict_noise else (pred - x0) ** 2
        loss = loss * self.loss_weight * (1 - self.fix_mask)

        if left_anchor.any() or right_anchor.any():
            anchor_mask = torch.ones_like(loss)
            if left_anchor.any():
                anchor_mask[left_anchor, 0, :] = 0.0
            if right_anchor.any():
                anchor_mask[right_anchor, -1, :] = 0.0
            loss = loss * anchor_mask

        weighted_regression_tensor = kwargs.get("weighted_regression_tensor", None)
        if weighted_regression_tensor is not None:
            loss = loss * weighted_regression_tensor.unsqueeze(-1)

        return loss.mean()

    def _prepare_sampler_tensors(
        self,
        schedule: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alphas = self.alpha[schedule]
        sigmas = self.sigma[schedule]
        stds = torch.zeros_like(alphas)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()
        return alphas, sigmas, stds

    def sample_interleaved_segments(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        num_segments: int,
        solver: str = "ddim",
        sample_steps: int = 50,
        sample_step_schedule: Union[str, Callable] = "uniform",
        use_ema: bool = True,
        temperature: float = 1.0,
        condition_cg=None,
        w_cg: float = 0.0,
        overlap_score_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        model = self.model_ema if use_ema else self.model
        start_state = start_state.to(self.device)
        goal_state = goal_state.to(self.device)
        batch_size, obs_dim = start_state.shape
        horizon = self.segment_horizon
        schedule = self._resolve_schedule(sample_steps, sample_step_schedule)
        alphas, sigmas, stds = self._prepare_sampler_tensors(schedule)

        segments = [
            torch.randn((batch_size, horizon, obs_dim), device=self.device, dtype=start_state.dtype) * temperature
            for _ in range(num_segments)
        ]

        for i in reversed(range(1, sample_steps + 1)):
            current_step = int(schedule[i].item())
            prev_step = int(schedule[i - 1].item())
            alpha_t = at_least_ndim(alphas[i], segments[0].dim())
            sigma_t = at_least_ndim(sigmas[i], segments[0].dim())
            alpha_prev = at_least_ndim(alphas[i - 1], segments[0].dim())
            sigma_prev = at_least_ndim(sigmas[i - 1], segments[0].dim())
            std_t = at_least_ndim(stds[i], segments[0].dim())

            for seg_idx in range(num_segments):
                current = segments[seg_idx]
                left_mode = "drop"
                right_mode = "drop"
                left_chunk = None
                right_chunk = None
                left_step = None
                right_step = None
                anchor_start = None
                anchor_goal = None

                if seg_idx == 0:
                    left_mode = "anchor"
                    anchor_start = start_state
                else:
                    left_mode = "overlap"
                    left_chunk = segments[seg_idx - 1][:, -self.overlap_steps :, :].detach()
                    left_step = torch.full((batch_size,), prev_step, device=self.device, dtype=torch.long)

                if seg_idx == num_segments - 1:
                    right_mode = "anchor"
                    anchor_goal = goal_state
                else:
                    right_mode = "overlap"
                    right_chunk = segments[seg_idx + 1][:, : self.overlap_steps, :].detach()
                    right_step = torch.full((batch_size,), current_step, device=self.device, dtype=torch.long)

                current = self._apply_anchor(current, anchor_start, anchor_goal)
                payload = self._make_condition_payload(
                    batch_size=batch_size,
                    obs_dim=obs_dim,
                    dtype=current.dtype,
                    device=current.device,
                    left_mode=left_mode,
                    right_mode=right_mode,
                    left_chunk=left_chunk,
                    right_chunk=right_chunk,
                    left_step=left_step,
                    right_step=right_step,
                )

                segments[seg_idx] = self._single_reverse_step(
                    current,
                    current_step,
                    prev_step,
                    alpha_t,
                    sigma_t,
                    alpha_prev,
                    sigma_prev,
                    std_t,
                    solver,
                    model,
                    payload,
                    condition_cg,
                    w_cg,
                )

        segments[0] = self._apply_anchor(segments[0], start_state=start_state)
        segments[-1] = self._apply_anchor(segments[-1], goal_state=goal_state)
        overlap_scores = self._compute_overlap_mismatch_scores(segments, scale=overlap_score_scale)
        order = torch.argsort(overlap_scores)
        segments = [segment[order] for segment in segments]
        overlap_scores = overlap_scores[order].detach().cpu().numpy()
        return segments, {"candidate_scores": overlap_scores, "score_kind": "overlap_mismatch"}

    def _average_overlap_in_place(self, segments: List[torch.Tensor]) -> None:
        for idx in range(1, len(segments)):
            left = segments[idx - 1][:, -self.overlap_steps :, :]
            right = segments[idx][:, : self.overlap_steps, :]
            fused = 0.5 * (left + right)
            segments[idx - 1][:, -self.overlap_steps :, :] = fused
            segments[idx][:, : self.overlap_steps, :] = fused

    def _compute_overlap_mismatch_scores(
        self,
        segments: List[torch.Tensor],
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(segments) <= 1:
            return torch.zeros((segments[0].shape[0],), device=segments[0].device, dtype=segments[0].dtype)

        scaled = None
        if scale is not None:
            scaled = scale.to(device=segments[0].device, dtype=segments[0].dtype).reshape(1, 1, -1)

        pairwise = []
        for idx in range(len(segments) - 1):
            diff = segments[idx][:, -self.overlap_steps :, :] - segments[idx + 1][:, : self.overlap_steps, :]
            if scaled is not None:
                diff = diff * scaled
            pairwise.append((diff ** 2).mean(dim=(1, 2)))
        return torch.stack(pairwise, dim=1).sum(dim=1)

    def _compute_inversion_score(
        self,
        x: torch.Tensor,
        model,
        payload: Dict[str, torch.Tensor],
        sample_steps: int,
    ) -> torch.Tensor:
        window = max(2, int(sample_steps * self.gsc_inversion_ratio))
        upper = min(self.diffusion_steps - 2, window)
        preds = []
        x_forward = x.clone()

        for t_idx in range(1, upper + 1):
            t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
            alpha_t = at_least_ndim(self.alpha[t_idx], x.dim())
            sigma_t = at_least_ndim(self.sigma[t_idx], x.dim())
            pred = self._predict_with_context(
                x_forward,
                t,
                alpha_t,
                sigma_t,
                model,
                payload,
                condition_cg=None,
                w_cg=0.0,
            )
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(x_forward, alpha_t, sigma_t, pred)
            x0_theta = pred if not self.predict_noise else epstheta_to_xtheta(x_forward, alpha_t, sigma_t, pred)
            x0_theta = x0_theta.clamp(-1.0, 1.0)

            alpha_next = at_least_ndim(self.alpha[t_idx + 1], x.dim())
            sigma_next = at_least_ndim(self.sigma[t_idx + 1], x.dim())
            x_forward = alpha_next * x0_theta + sigma_next * eps_theta
            preds.append(eps_theta)

        if len(preds) < 2:
            return torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)

        noise_path = torch.stack(preds, dim=1)
        derivative = torch.diff(noise_path, dim=1)
        return torch.linalg.vector_norm(derivative.reshape(x.shape[0], -1), dim=1)

    def sample_gsc_segments(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        num_segments: int,
        solver: str = "ddim",
        sample_steps: int = 50,
        sample_step_schedule: Union[str, Callable] = "uniform",
        use_ema: bool = True,
        temperature: float = 1.0,
        condition_cg=None,
        w_cg: float = 0.0,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        model = self.model_ema if use_ema else self.model
        start_state = start_state.to(self.device)
        goal_state = goal_state.to(self.device)
        batch_size, obs_dim = start_state.shape
        horizon = self.segment_horizon
        schedule = self._resolve_schedule(sample_steps, sample_step_schedule)
        alphas, sigmas, stds = self._prepare_sampler_tensors(schedule)

        segments = [
            torch.randn((batch_size, horizon, obs_dim), device=self.device, dtype=start_state.dtype) * temperature
            for _ in range(num_segments)
        ]
        self.latest_gsc_scores = None
        latest_score_stack = None

        late_window = int(math.ceil((1.0 - self.gsc_filter_start) * sample_steps))
        late_window = min(max(late_window, 1), sample_steps)

        for i in reversed(range(1, sample_steps + 1)):
            current_step = int(schedule[i].item())
            prev_step = int(schedule[i - 1].item())
            alpha_t = at_least_ndim(alphas[i], segments[0].dim())
            sigma_t = at_least_ndim(sigmas[i], segments[0].dim())
            alpha_prev = at_least_ndim(alphas[i - 1], segments[0].dim())
            sigma_prev = at_least_ndim(sigmas[i - 1], segments[0].dim())
            std_t = at_least_ndim(stds[i], segments[0].dim())

            inversion_scores = None
            inner_loops = max(1, self.gsc_inner_loops)
            for inner_idx in range(inner_loops):
                self._average_overlap_in_place(segments)
                updated_segments = []
                current_scores = []

                for seg_idx in range(num_segments):
                    anchor_start = start_state if seg_idx == 0 else None
                    anchor_goal = goal_state if seg_idx == num_segments - 1 else None
                    current = self._apply_anchor(segments[seg_idx], anchor_start, anchor_goal)

                    payload = self._make_condition_payload(
                        batch_size=batch_size,
                        obs_dim=obs_dim,
                        dtype=current.dtype,
                        device=current.device,
                        left_mode="anchor" if seg_idx == 0 else "drop",
                        right_mode="anchor" if seg_idx == num_segments - 1 else "drop",
                    )

                    denoised = self._single_reverse_step(
                        current,
                        current_step,
                        prev_step,
                        alpha_t,
                        sigma_t,
                        alpha_prev,
                        sigma_prev,
                        std_t,
                        solver,
                        model,
                        payload,
                        condition_cg,
                        w_cg,
                    )

                    if inner_idx == inner_loops - 1 and i <= late_window:
                        current_scores.append(self._compute_inversion_score(denoised, model, payload, sample_steps))

                    if inner_idx < inner_loops - 1 and prev_step > 0:
                        denoised = self._forward_jump(denoised, alpha_prev, sigma_prev, alpha_t, sigma_t)

                    updated_segments.append(denoised)

                segments = updated_segments
                if current_scores:
                    inversion_scores = current_scores

            if inversion_scores is not None:
                latest_score_stack = torch.stack(inversion_scores, dim=1)

        self._average_overlap_in_place(segments)
        segments[0] = self._apply_anchor(segments[0], start_state=start_state)
        segments[-1] = self._apply_anchor(segments[-1], goal_state=goal_state)
        candidate_scores = None
        if latest_score_stack is not None:
            mean_score = latest_score_stack.mean(dim=1)
            order = torch.argsort(mean_score)
            segments = [segment[order] for segment in segments]
            latest_score_stack = latest_score_stack[order]
            self.latest_gsc_scores = [
                latest_score_stack[:, idx].detach().cpu().numpy() for idx in range(latest_score_stack.shape[1])
            ]
            candidate_scores = mean_score[order].detach().cpu().numpy()
        return segments, {"gsc_scores": self.latest_gsc_scores, "candidate_scores": candidate_scores, "score_kind": "gsc"}

    def sample_segment_chain(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        num_segments: int,
        strategy: str,
        solver: str = "ddim",
        sample_steps: int = 50,
        sample_step_schedule: Union[str, Callable] = "uniform",
        use_ema: bool = True,
        temperature: float = 1.0,
        condition_cg=None,
        w_cg: float = 0.0,
        overlap_score_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        strategy = str(strategy).lower()
        if strategy == "interleave":
            return self.sample_interleaved_segments(
                start_state=start_state,
                goal_state=goal_state,
                num_segments=num_segments,
                solver=solver,
                sample_steps=sample_steps,
                sample_step_schedule=sample_step_schedule,
                use_ema=use_ema,
                temperature=temperature,
                condition_cg=condition_cg,
                w_cg=w_cg,
                overlap_score_scale=overlap_score_scale,
            )
        if strategy == "gsc":
            return self.sample_gsc_segments(
                start_state=start_state,
                goal_state=goal_state,
                num_segments=num_segments,
                solver=solver,
                sample_steps=sample_steps,
                sample_step_schedule=sample_step_schedule,
                use_ema=use_ema,
                temperature=temperature,
                condition_cg=condition_cg,
                w_cg=w_cg,
            )
        raise ValueError(f"Unsupported stitch strategy: {strategy}")

    @staticmethod
    def rank_overlap_mismatch(segments_np: List[np.ndarray], overlap_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        pairwise = []
        for idx in range(len(segments_np) - 1):
            left_tail = segments_np[idx][:, -overlap_steps:, :]
            right_head = segments_np[idx + 1][:, :overlap_steps, :]
            pairwise.append(((left_tail - right_head) ** 2).mean(axis=(1, 2)))
        stacked = np.stack(pairwise, axis=1)
        scores = stacked.sum(axis=1)
        order = np.argsort(scores)
        return order, scores

    @staticmethod
    def _blend_weights(length: int, mode: str, beta: float) -> np.ndarray:
        if mode in ["avg", "mean"]:
            return np.full((length,), 0.5, dtype=np.float32)

        if length == 1:
            return np.ones((1,), dtype=np.float32)

        t = np.arange(length, dtype=np.float32)
        last = float(length - 1)
        if mode in ["exp", "exponential"]:
            beta = max(float(beta), 1e-6)
            weights = (np.exp(-beta * t / last) - np.exp(-beta)) / (1.0 - np.exp(-beta))
        elif mode == "cosine":
            weights = 0.5 * (1.0 + np.cos(np.pi * t / last))
        elif mode == "linear":
            weights = 1.0 - t / last
        elif mode == "smoothstep":
            x = t / last
            weights = 1.0 - (3.0 * x * x - 2.0 * x * x * x)
        else:
            raise ValueError(f"Unsupported overlap blend mode: {mode}")
        return weights.astype(np.float32)

    @classmethod
    def blend_segments(
        cls,
        segments_np: List[np.ndarray],
        overlap_steps: int,
        blend_mode: str,
        blend_beta: float,
        target_horizon: Optional[int] = None,
    ) -> np.ndarray:
        num_segments = len(segments_np)
        batch_size, seg_horizon, obs_dim = segments_np[0].shape
        step_size = seg_horizon - overlap_steps
        total_horizon = num_segments * seg_horizon - (num_segments - 1) * overlap_steps
        stitched = np.zeros((batch_size, total_horizon, obs_dim), dtype=np.float32)

        stitched[:, :step_size, :] = segments_np[0][:, :step_size, :]
        for idx in range(1, num_segments - 1):
            begin = seg_horizon + (idx - 1) * step_size
            end = begin + (seg_horizon - 2 * overlap_steps)
            stitched[:, begin:end, :] = segments_np[idx][:, overlap_steps:-overlap_steps, :]
        stitched[:, total_horizon - step_size :, :] = segments_np[-1][:, overlap_steps:, :]

        weights = cls._blend_weights(overlap_steps, blend_mode, blend_beta)[None, :, None]
        for idx in range(num_segments - 1):
            begin = (idx + 1) * step_size
            end = begin + overlap_steps
            left_tail = segments_np[idx][:, -overlap_steps:, :]
            right_head = segments_np[idx + 1][:, :overlap_steps, :]
            stitched[:, begin:end, :] = weights * left_tail + (1.0 - weights) * right_head

        if target_horizon is not None:
            stitched = stitched[:, :target_horizon, :]
        return stitched
