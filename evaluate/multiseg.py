import math
from collections import defaultdict

import numpy as np
import torch
from tqdm import trange

from .base import add_to, flatten, max_episode_lengths, object_num
from pipelines.utils import resolve_goal_indices


def _resolve_num_segments(total_horizon, segment_horizon, overlap_length, configured_num_segments):
    if configured_num_segments is not None and configured_num_segments > 0:
        return int(configured_num_segments)

    stride = max(1, segment_horizon - overlap_length)
    if total_horizon <= segment_horizon:
        return 2
    return int(math.ceil((total_horizon - segment_horizon) / stride) + 1)


def multi_segment_evaluate(
    diffusions_model,
    mode,
    low_controller,
    env,
    normalizer,
    task_id,
    horizon,
    obs_dim,
    act_dim,
    config,
    num_eval_episodes,
    num_video_episodes,
    video_frame_skip,
):
    """Evaluate with multi-segment overlap stitching planner."""
    assert mode in ["plan_every_step", "achieve_subgoal"], "Invalid evaluation mode!"

    multi_cfg = config.multi_segment
    segment_horizon = int(multi_cfg.segment_horizon)
    overlap_length = int(multi_cfg.overlap_length)
    num_segments = _resolve_num_segments(
        total_horizon=horizon,
        segment_horizon=segment_horizon,
        overlap_length=overlap_length,
        configured_num_segments=multi_cfg.get("num_segments", None),
    )

    goal_indices = resolve_goal_indices(config.task, obs_dim)

    task_trajectories = []
    task_stats_collector = defaultdict(list)
    task_renders_list = []
    task_trajectories_3d = []
    task_segment_trajectories_3d = []
    task_actual_trajectories_3d = []

    for i_episode in trange(num_eval_episodes + num_video_episodes, desc=f"Task {task_id} Episodes", leave=False):
        current_episode_trajectory_data = defaultdict(list)
        is_video_episode = i_episode >= num_eval_episodes
        reset_options = {"task_id": task_id, "render_goal": is_video_episode}

        obs, info = env.reset(options=reset_options)
        task_overall_goal_state = info.get("goal")
        goal_frame_rendered_from_env = info.get("goal_rendered")

        episode_done = False
        current_episode_step = 0
        rendered_frames_for_this_episode = []

        current_plan_observations = None
        current_subgoal_obs = None
        current_subgoal_idx_in_plan = -1
        jump_steps = config.task.get("jump_steps", 1)
        plan_lookahead = max(1, config.task.low_horizon // jump_steps)

        current_episode_actual_object_traj = None
        if is_video_episode:
            object_num_in_env = object_num.get(config.task.env_name, 1)
            current_episode_actual_object_traj = [[] for _ in range(object_num_in_env)]
            for obj_idx in range(object_num_in_env):
                obj_pos = obs[:obs_dim][-9 * obj_idx - 9 : -9 * obj_idx - 6]
                current_episode_actual_object_traj[obj_idx].append(np.array(obj_pos, copy=True))

        while (not episode_done) and (current_episode_step < max_episode_lengths[config.task.env_name]):
            normalized_current_obs = normalizer.normalize(obs[:obs_dim][None])
            normalized_task_goal = normalizer.normalize(task_overall_goal_state[:obs_dim][None])

            need_replan = False
            if current_episode_step == 0:
                need_replan = True
            elif config.task.replan_every > 0 and current_episode_step % config.task.replan_every == 0:
                need_replan = True
            elif mode == "plan_every_step":
                need_replan = True
            elif mode == "achieve_subgoal" and current_subgoal_obs is not None:
                dist_to_subgoal = np.linalg.norm((obs[:obs_dim] - current_subgoal_obs)[goal_indices])
                if dist_to_subgoal > config.task.get("replan_error_threshold", 3.0):
                    need_replan = True

            if need_replan:
                start_state = torch.tensor(
                    normalized_current_obs,
                    device=config.device,
                    dtype=torch.float32,
                ).repeat(config.planner_num_candidates, 1)
                goal_state = torch.tensor(
                    normalized_task_goal,
                    device=config.device,
                    dtype=torch.float32,
                ).repeat(config.planner_num_candidates, 1)

                if config.enable_distance_guidance:
                    condition_cg = goal_state.unsqueeze(1).repeat(1, segment_horizon, 1)
                    w_cg = multi_cfg.get("w_cg", 0.5)
                else:
                    condition_cg = None
                    w_cg = 0.0

                traj_normalized, sample_log = diffusions_model.sample_multi_segment(
                    start_state=start_state,
                    goal_state=goal_state,
                    num_segments=num_segments,
                    segment_horizon=segment_horizon,
                    overlap_length=overlap_length,
                    inner_resample_rounds=int(multi_cfg.inner_resample_rounds),
                    overlap_blend_type=str(multi_cfg.get("overlap_blend_type", "avg")),
                    overlap_exp_beta=float(multi_cfg.get("overlap_exp_beta", 3.0)),
                    repaint_times=int(multi_cfg.repaint_times),
                    jump_len=int(multi_cfg.jump_len),
                    solver=config.planner_solver,
                    sample_steps=int(config.planner_sampling_steps),
                    use_ema=config.planner_use_ema,
                    temperature=float(config.task.planner_temperature),
                    condition_cg=condition_cg,
                    w_cg=w_cg,
                )

                overlap_cost = sample_log.get("candidate_overlap_cost", None)
                if overlap_cost is None:
                    overlap_cost_np = np.zeros((traj_normalized.shape[0],), dtype=np.float32)
                else:
                    overlap_cost_np = overlap_cost.detach().cpu().numpy().astype(np.float32)

                sorted_idx = np.argsort(overlap_cost_np)
                top_n = int(multi_cfg.get("top_n_candidates", 1))
                top_n = max(1, min(top_n, sorted_idx.shape[0]))
                top_idx = sorted_idx[:top_n]

                pick_mode = str(multi_cfg.get("pick_mode", "first"))
                if pick_mode == "first":
                    idx = int(top_idx[0])
                elif pick_mode == "rand":
                    idx = int(np.random.choice(top_idx))
                elif pick_mode == "score_weighted":
                    temp = float(multi_cfg.get("score_temperature", 5.0))
                    rel = overlap_cost_np[top_idx] - float(overlap_cost_np[top_idx].min())
                    w = np.exp(-temp * rel)
                    w = w / (w.sum() + 1e-8)
                    idx = int(np.random.choice(top_idx, p=w))
                else:
                    raise ValueError(f"Unsupported pick_mode: {pick_mode}")

                current_plan_observations = normalizer.unnormalize(traj_normalized[idx].cpu().numpy())
                current_subgoal_idx_in_plan = min(plan_lookahead, current_plan_observations.shape[0] - 1)
                current_subgoal_obs = current_plan_observations[current_subgoal_idx_in_plan, :]

                if current_episode_step == 0 and is_video_episode:
                    object_num_in_env = object_num.get(config.task.env_name, 1)

                    segment_states = sample_log.get("segments", None)
                    if segment_states is not None and len(segment_states) > 0:
                        segment_traj_3d = np.zeros((len(segment_states), object_num_in_env, segment_states[0].shape[1], 3))
                        for seg_i, seg_state in enumerate(segment_states):
                            seg_obs = normalizer.unnormalize(seg_state[idx].cpu().numpy())
                            for obj_i in range(object_num_in_env):
                                segment_traj_3d[seg_i, obj_i] = np.array(seg_obs[:, -9 * obj_i - 9 : -9 * obj_i - 6])
                        segment_traj_3d[-1, :, -1, :] = np.stack(
                            [task_overall_goal_state[:obs_dim][-9 * obj_i - 9 : -9 * obj_i - 6] for obj_i in range(object_num_in_env)],
                            axis=0,
                        )
                        task_segment_trajectories_3d.append(segment_traj_3d)

                    traj_3d = np.zeros((object_num_in_env, current_plan_observations.shape[0], 3))
                    for obj_i in range(object_num_in_env):
                        traj_3d[obj_i] = np.array(current_plan_observations[:, -9 * obj_i - 9 : -9 * obj_i - 6])
                        traj_3d[obj_i, -1, :] = task_overall_goal_state[:obs_dim][-9 * obj_i - 9 : -9 * obj_i - 6]
                    task_trajectories_3d.append(traj_3d)

            elif mode == "achieve_subgoal":
                distance_to_subgoal = np.linalg.norm((obs[:obs_dim] - current_subgoal_obs)[goal_indices])
                if distance_to_subgoal <= config.task.goal_tol:
                    current_subgoal_idx_in_plan = min(
                        current_subgoal_idx_in_plan + plan_lookahead,
                        current_plan_observations.shape[0] - 1,
                    )
                    current_subgoal_obs = current_plan_observations[current_subgoal_idx_in_plan, :]

            if config.use_diffusion_invdyn:
                policy_prior = torch.zeros((1, act_dim), device=config.device)
                with torch.no_grad():
                    next_obs_plan = normalizer.normalize(current_subgoal_obs)
                    obs_policy = torch.tensor(normalized_current_obs, device=config.device, dtype=torch.float32)
                    next_obs_policy = torch.tensor(next_obs_plan, device=config.device, dtype=torch.float32)
                    if obs_policy.dim() == 1:
                        obs_policy = obs_policy.unsqueeze(0)
                    if next_obs_policy.dim() == 1:
                        next_obs_policy = next_obs_policy.unsqueeze(0)
                    act, _ = low_controller.sample(
                        policy_prior,
                        solver=config.policy_solver,
                        n_samples=1,
                        sample_steps=config.policy_sampling_steps,
                        condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                        w_cfg=1.0,
                        use_ema=config.policy_use_ema,
                        temperature=config.policy_temperature,
                    )
                    action = act.clip(-1.0, 1.0).squeeze().cpu().numpy()
            else:
                normalized_subgoal_obs_for_low = normalizer.normalize(current_subgoal_obs)[goal_indices]
                with torch.no_grad():
                    action = (
                        low_controller.predict(normalized_current_obs, normalized_subgoal_obs_for_low)
                        .clip(-1.0, 1.0)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_done = info["success"]
            current_episode_step += 1

            if is_video_episode and current_episode_actual_object_traj is not None:
                for obj_idx in range(len(current_episode_actual_object_traj)):
                    obj_pos = next_obs[:obs_dim][-9 * obj_idx - 9 : -9 * obj_idx - 6]
                    current_episode_actual_object_traj[obj_idx].append(np.array(obj_pos, copy=True))

            if is_video_episode and (current_episode_step % video_frame_skip == 0 or episode_done):
                frame = env.render().copy()
                if goal_frame_rendered_from_env is not None:
                    rendered_frames_for_this_episode.append(np.concatenate([goal_frame_rendered_from_env, frame], axis=0))
                else:
                    rendered_frames_for_this_episode.append(frame)

            if not is_video_episode:
                transition_payload = dict(
                    observation=obs.copy(),
                    next_observation=next_obs.copy(),
                    action=action.copy(),
                    reward=reward,
                    done=bool(terminated or truncated),
                    info=info.copy(),
                )
                add_to(current_episode_trajectory_data, transition_payload)

            obs = next_obs

        if not is_video_episode:
            task_trajectories.append(dict(current_episode_trajectory_data))
            add_to(task_stats_collector, flatten(info))
        else:
            if rendered_frames_for_this_episode:
                task_renders_list.append(np.array(rendered_frames_for_this_episode))
            if current_episode_actual_object_traj is not None:
                task_actual_trajectories_3d.append(
                    np.stack([np.stack(points, axis=0) for points in current_episode_actual_object_traj], axis=0)
                )

    aggregated_stats_for_this_task = {}
    for k, v_list in task_stats_collector.items():
        numeric_vals = [x for x in v_list if isinstance(x, (int, float, np.number))]
        if numeric_vals:
            aggregated_stats_for_this_task[k] = np.mean(numeric_vals)
        elif v_list:
            aggregated_stats_for_this_task[k] = v_list

    return (
        aggregated_stats_for_this_task,
        task_trajectories,
        task_renders_list,
        task_trajectories_3d,
        task_segment_trajectories_3d,
        task_actual_trajectories_3d,
    )
