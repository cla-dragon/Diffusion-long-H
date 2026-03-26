import math
from collections import defaultdict

import numpy as np
import torch
from tqdm import trange

from .base import add_to, max_episode_lengths, object_num
from pipelines.utils import resolve_goal_indices


def _resolve_num_segments(total_horizon, segment_horizon, overlap_steps, configured_num_segments):
    if configured_num_segments is not None and configured_num_segments > 0:
        return int(configured_num_segments)

    stride = max(1, segment_horizon - overlap_steps)
    if total_horizon <= segment_horizon:
        return 2
    return int(math.ceil((total_horizon - segment_horizon) / stride) + 1)


def segment_stitch_evaluate(
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
    assert mode in ["plan_every_step", "achieve_subgoal"], "Invalid evaluation mode!"

    stitch_cfg = config.stitch
    segment_horizon = int(stitch_cfg.segment_model_horizon)
    overlap_steps = int(stitch_cfg.overlap_steps)
    num_segments = _resolve_num_segments(
        total_horizon=horizon,
        segment_horizon=segment_horizon,
        overlap_steps=overlap_steps,
        configured_num_segments=stitch_cfg.get("num_segments", None),
    )
    goal_indices = resolve_goal_indices(config.task, obs_dim)

    task_trajectories = []
    task_stats_collector = defaultdict(list)
    task_renders_list = []
    task_trajectories_3d = []
    task_actual_trajectories_3d = []

    for i_episode in trange(num_eval_episodes + num_video_episodes, desc=f"Task {task_id} Episodes", leave=False):
        current_episode_trajectory_data = defaultdict(list)
        is_video_episode = i_episode >= num_eval_episodes
        reset_options = {"task_id": task_id, "render_goal": is_video_episode}

        obs, info = env.reset(options=reset_options)
        task_goal_state = info.get("goal")
        goal_frame = info.get("goal_rendered")

        episode_done = False
        current_episode_step = 0
        rendered_frames = []

        current_plan = None
        current_subgoal = None
        current_subgoal_idx = -1
        jump_steps = config.task.get("jump_steps", 1)
        plan_lookahead = max(1, config.task.low_horizon // jump_steps)

        current_episode_actual_object_traj = None
        if is_video_episode:
            obj_count = object_num.get(config.task.env_name, 1)
            current_episode_actual_object_traj = [[] for _ in range(obj_count)]
            for obj_idx in range(obj_count):
                obj_pos = obs[:obs_dim][-9 * obj_idx - 9 : -9 * obj_idx - 6]
                current_episode_actual_object_traj[obj_idx].append(np.array(obj_pos, copy=True))

        while (not episode_done) and (current_episode_step < max_episode_lengths[config.task.env_name]):
            normalized_obs = normalizer.normalize(obs[:obs_dim][None])
            normalized_goal = normalizer.normalize(task_goal_state[:obs_dim][None])

            need_replan = False
            if current_episode_step == 0:
                need_replan = True
            elif config.task.replan_every > 0 and current_episode_step % config.task.replan_every == 0:
                need_replan = True
            elif mode == "plan_every_step":
                need_replan = True
            elif mode == "achieve_subgoal" and current_subgoal is not None:
                subgoal_error = np.linalg.norm((obs[:obs_dim] - current_subgoal)[goal_indices])
                if subgoal_error > config.task.get("replan_error_threshold", 3.0):
                    need_replan = True

            if need_replan:
                start_state = torch.as_tensor(
                    normalized_obs,
                    device=config.device,
                    dtype=torch.float32,
                ).repeat(config.planner_num_candidates, 1)
                goal_state = torch.as_tensor(
                    normalized_goal,
                    device=config.device,
                    dtype=torch.float32,
                ).repeat(config.planner_num_candidates, 1)

                if config.enable_distance_guidance:
                    condition_cg = goal_state.unsqueeze(1).repeat(1, segment_horizon, 1)
                    w_cg = float(config.task.get("w_ldg", 0.5))
                else:
                    condition_cg = None
                    w_cg = 0.0

                segment_list, sample_log = diffusions_model.sample_segment_chain(
                    start_state=start_state,
                    goal_state=goal_state,
                    num_segments=num_segments,
                    strategy=str(stitch_cfg.strategy),
                    solver=config.planner_solver,
                    sample_steps=int(config.planner_sampling_steps),
                    sample_step_schedule=str(config.planner_sampling_schedule),
                    use_ema=config.planner_use_ema,
                    temperature=float(config.task.planner_temperature),
                    condition_cg=condition_cg,
                    w_cg=w_cg,
                )

                segment_list_np = [
                    normalizer.unnormalize(segment.detach().cpu().numpy()) for segment in segment_list
                ]
                candidate_order, candidate_scores = diffusions_model.rank_overlap_mismatch(
                    segment_list_np,
                    overlap_steps=overlap_steps,
                )

                top_n = max(1, min(int(stitch_cfg.top_n), candidate_order.shape[0]))
                chosen_pool = candidate_order[:top_n]

                top_segments = [segment[chosen_pool] for segment in segment_list_np]
                stitched_candidates = diffusions_model.blend_segments(
                    top_segments,
                    overlap_steps=overlap_steps,
                    blend_mode=str(stitch_cfg.merge_mode),
                    blend_beta=float(stitch_cfg.merge_beta),
                    target_horizon=horizon,
                )

                if stitch_cfg.pick_mode == "first":
                    chosen_idx = 0
                elif stitch_cfg.pick_mode == "rand":
                    chosen_idx = int(np.random.randint(top_n))
                else:
                    raise ValueError(f"Unsupported pick_mode: {stitch_cfg.pick_mode}")

                selected_score = float(candidate_scores[chosen_pool[chosen_idx]])
                current_plan = stitched_candidates[chosen_idx]
                current_subgoal_idx = min(plan_lookahead, current_plan.shape[0] - 1)
                current_subgoal = current_plan[current_subgoal_idx]

                if current_episode_step == 0 and is_video_episode:
                    obj_count = object_num.get(config.task.env_name, 1)
                    plan_xyz = np.zeros((obj_count, current_plan.shape[0], 3), dtype=np.float32)
                    for obj_idx in range(obj_count):
                        plan_xyz[obj_idx] = np.asarray(current_plan[:, -9 * obj_idx - 9 : -9 * obj_idx - 6], dtype=np.float32)
                        plan_xyz[obj_idx, -1, :] = task_goal_state[:obs_dim][-9 * obj_idx - 9 : -9 * obj_idx - 6]
                    task_trajectories_3d.append(plan_xyz)

                task_stats_collector["selected_overlap_score"].append(selected_score)

                gsc_scores = sample_log.get("gsc_scores", None)
                if gsc_scores is not None:
                    per_segment = np.asarray([float(scores[chosen_pool[chosen_idx]]) for scores in gsc_scores], dtype=np.float32)
                    task_stats_collector["selected_gsc_score"].append(float(per_segment.mean()))

            elif mode == "achieve_subgoal":
                distance_to_subgoal = np.linalg.norm((obs[:obs_dim] - current_subgoal)[goal_indices])
                if distance_to_subgoal <= config.task.goal_tol:
                    current_subgoal_idx = min(current_subgoal_idx + plan_lookahead, current_plan.shape[0] - 1)
                    current_subgoal = current_plan[current_subgoal_idx]

            if config.use_diffusion_invdyn:
                policy_prior = torch.zeros((1, act_dim), device=config.device)
                with torch.no_grad():
                    next_obs_plan = normalizer.normalize(current_subgoal)
                    obs_policy = torch.as_tensor(normalized_obs, device=config.device, dtype=torch.float32)
                    next_obs_policy = torch.as_tensor(next_obs_plan, device=config.device, dtype=torch.float32)
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
                normalized_subgoal = normalizer.normalize(current_subgoal)[goal_indices]
                with torch.no_grad():
                    action = (
                        low_controller.predict(normalized_obs, normalized_subgoal)
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
                if goal_frame is not None:
                    rendered_frames.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    rendered_frames.append(frame)

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
            success = 1.0 if episode_done else 0.0
            task_stats_collector["success"].append(success)
            task_trajectories.append(current_episode_trajectory_data)
        else:
            if rendered_frames:
                task_renders_list.append(np.asarray(rendered_frames))
            if current_episode_actual_object_traj is not None:
                task_actual_trajectories_3d.append(
                    np.asarray([np.asarray(traj, dtype=np.float32) for traj in current_episode_actual_object_traj], dtype=np.float32)
                )

    eval_info = {key: float(np.mean(value)) for key, value in task_stats_collector.items() if len(value) > 0}
    return eval_info, task_trajectories, task_renders_list, task_trajectories_3d, task_actual_trajectories_3d
