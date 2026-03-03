import numpy as np
import torch

from collections import defaultdict
from tqdm import trange

max_episode_lengths = {
    "scene-play-v0": 750,
    "cube-single-play-v0": 200,
    "cube-double-play-v0": 500,
    "antsoccer-arena-stitch-v0": 1000,
    "antsoccer-medium-navigate-v0": 1000
}

object_num = {
    "cube-single-play-v0": 1,
    "cube-double-play-v0": 2,
    "cube-triple-play-v0": 3,
}

def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict): # Check if it's a dict
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def single_layer_evaluate(
    diffusions_model,
    mode, # evaluation mode: how to generate actions
    low_controller,
    env,
    normalizer,
    task_id, # The identifier the environment's reset() function expects
    horizon, # planning horizon which is reduced by jump_steps
    obs_dim,
    act_dim,
    config, # Your Hydra args object
    num_eval_episodes,
    num_video_episodes,
    video_frame_skip,
):
    
    """
    Evaluates the single layer agent(not hierarchical) on a single task.
    """
    assert mode in ['plan_every_step', 'achieve_subgoal'], "Invalid evaluation mode!"
    if config.adaptive_replan_horizon and mode != 'achieve_subgoal':
        print("Warning: adaptive_replan_horizon is enabled but mode is not 'achieve_subgoal'. Setting mode to 'achieve_subgoal'.")
        mode = 'achieve_subgoal'

    # Templates for planning priors
    prior_template = torch.zeros((1, horizon, obs_dim), device=config.device)
    if config.plan_only_object_goal:
        goal_dim = 9 * object_num.get(config.task.env_name, 1)
    else:
        goal_dim = obs_dim

    task_trajectories = []
    task_stats_collector = defaultdict(list)
    task_renders_list = []
    task_trajectories_3d = []

    for i_episode in trange(num_eval_episodes + num_video_episodes, desc=f"Task {task_id} Episodes", leave=False):
        current_episode_trajectory_data = defaultdict(list)
        is_video_episode = i_episode >= num_eval_episodes
        reset_options = {'task_id': task_id}
        reset_options['render_goal'] = is_video_episode

        obs, info = env.reset(options=reset_options)
        task_overall_goal_state = info.get('goal') # Final goal for the entire task from environment
        goal_frame_rendered_from_env = info.get('goal_rendered') # For video rendering

        episode_done = False
        current_episode_step = 0
        rendered_frames_for_this_episode = []
        
        current_plan_observations = None 
        current_subgoal_obs = None   
        current_subgoal_idx_in_plan = -1
        current_finshed_horizon = 0 # used for adaptive replan horizon
        temp_plan_valid_horizon = horizon - current_finshed_horizon # Record valid horizon for current plan

        while (not episode_done) and (current_episode_step < max_episode_lengths[config.task.env_name]):
            normalized_current_obs = normalizer.normalize(obs[:obs_dim][None])
            normalized_task_goal = normalizer.normalize(task_overall_goal_state[:obs_dim][None]) # [1, obs_dim]

            # Planning Step
            need_replan = False
            # Condition 1: Start
            if current_episode_step == 0:
                need_replan = True
            # Condition 2: Periodic replan
            elif config.task.replan_every > 0 and current_episode_step % config.task.replan_every == 0:
                need_replan = True
            # Condition 3: Plan every step mode
            elif mode == 'plan_every_step':
                need_replan = True
            # Condition 4: Adaptive Replan if deviation from subgoal is too large
            elif mode == 'achieve_subgoal' and config.get('adaptive_replan_on_error', False) and current_subgoal_obs is not None:
                dist_to_subgoal = np.linalg.norm((obs[:obs_dim] - current_subgoal_obs)[-goal_dim:])

                # Dynamically determine valid threshold (since max distance varies by env)
                threshold = config.task.get('replan_error_threshold', 2.0)
                if dist_to_subgoal > threshold:
                    print(f"Adaptive Replan Triggered: Deviation {dist_to_subgoal:.4f} > Threshold {threshold}")
                    need_replan = True

            if need_replan:
                # Create priors for planning
                current_prior = prior_template.clone()
                current_prior[:, 0] = torch.tensor(normalized_current_obs, device=config.device, dtype=torch.float32)
                if (not config.enable_distance_guidance) and (not config.adaptive_replan_horizon):
                    current_prior[:, -1] = torch.tensor(normalized_task_goal, device=config.device, dtype=torch.float32)

                # Planning
                if config.enable_distance_guidance:
                    condition_cg = torch.tensor(normalized_task_goal, device=config.device, dtype=torch.float32).unsqueeze(1) # [1, 1, obs_dim]
                    condition_cg = condition_cg.repeat(config.planner_num_candidates, horizon, 1)  # [num_candidates, horizon, obs_dim]
                    w_cg = 0.5
                else:
                    condition_cg = None
                    w_cg = 0.0
                
                # Adaptive Replan Horizon Adjustment
                if config.adaptive_replan_horizon:
                    adadptive_mask = torch.zeros((1, horizon, obs_dim))
                    adadptive_mask[0, 0, :] = 1.0 # Always condition on current obs
                    adadptive_mask[0, horizon - current_finshed_horizon - 1, :] = 1.0
                    current_prior[:, horizon - current_finshed_horizon - 1] = torch.tensor(normalized_task_goal, device=config.device, dtype=torch.float32)

                    traj_normalized, _ = diffusions_model.sample(
                        current_prior.repeat(config.planner_num_candidates, 1, 1),
                        mask=adadptive_mask, repaint_times=10, jump_len=10,
                        solver='ddpm', n_samples=config.planner_num_candidates,
                        sample_steps=200, use_ema=config.planner_use_ema,
                        temperature=config.task.planner_temperature, condition_cg=condition_cg, w_cg=w_cg
                    )

                else: # Normal fixed horizon mask
                    traj_normalized, _ = diffusions_model.sample(
                        current_prior.repeat(config.planner_num_candidates, 1, 1),
                        solver=config.planner_solver, n_samples=config.planner_num_candidates,
                        sample_steps=config.planner_sampling_steps, use_ema=config.planner_use_ema,
                        temperature=config.task.planner_temperature, condition_cg=condition_cg, w_cg=w_cg
                    )

                temp_plan_valid_horizon = horizon - current_finshed_horizon # Record valid horizon for current plan
                print(f"Replanning at step {current_episode_step}, valid horizon: {temp_plan_valid_horizon}.")

                # Select the best plan
                # TODO: N-step Q value estimator
                idx = 0
                current_plan_observations = normalizer.unnormalize(traj_normalized[idx, :, :].cpu().numpy())
                
                jump_steps = config.task.get('jump_steps', 1)
                plan_lookahead = max(1, config.task.low_horizon // jump_steps)
                
                current_subgoal_idx_in_plan = min(plan_lookahead, temp_plan_valid_horizon - 1)
                current_subgoal_obs = current_plan_observations[current_subgoal_idx_in_plan, :]
            
                if current_episode_step == 0 and is_video_episode:
                    # Visualize the planned 3D trajectory at the start of the episode
                    object_num_in_env = object_num.get(config.task.env_name, 1)
                    traj_3d = np.zeros((object_num_in_env, current_plan_observations.shape[0], 3))
                    for i in range(object_num.get(config.task.env_name, 1)):
                        traj_3d[i] = np.array(current_plan_observations[:, -9*i-9:-9*i-6])  # (T, 3)
                        traj_3d[i, -1, :] = task_overall_goal_state[:obs_dim][-9*i-9:-9*i-6]  # Set final point to goal
                    task_trajectories_3d.append(traj_3d)

            # Subgoal Selection and Update
            elif mode == 'achieve_subgoal':
                # Update subgoal if reached
                distance_to_subgoal = np.linalg.norm((obs[:obs_dim] - current_subgoal_obs)[-goal_dim:])
                if distance_to_subgoal <= config.task.goal_tol: 
                    jump_steps = config.task.get('jump_steps', 1)
                    plan_lookahead = max(1, config.task.low_horizon // jump_steps)

                    if current_subgoal_idx_in_plan == temp_plan_valid_horizon - 1:
                        print(f"Final subgoal reached at index {current_subgoal_idx_in_plan}. (Not finished yet.)")
                        current_subgoal_idx_in_plan = temp_plan_valid_horizon
                    else:
                        # Update finished horizon for adaptive replan
                        if config.adaptive_replan_horizon:
                            current_finshed_horizon = min(current_finshed_horizon + plan_lookahead, horizon - 1)
                            # if current_finshed_horizon == horizon - 1, then the entire task is finished!
                        
                        # current_finished_horizon will always be 0 if not adaptive replan
                        current_subgoal_idx_in_plan = min(current_subgoal_idx_in_plan + plan_lookahead, temp_plan_valid_horizon-1)
                        current_subgoal_obs = current_plan_observations[current_subgoal_idx_in_plan, :]
                        if config.adaptive_replan_horizon:
                            print(f"Subgoal reached, updated subgoal index to {current_subgoal_idx_in_plan}, finished horizon to {current_finshed_horizon}.")
                        else:
                            print(f"Subgoal reached, updating to index {current_subgoal_idx_in_plan}.")
                
            # Low-Level Control
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
                    act, log = low_controller.sample(
                                policy_prior,
                                solver=config.policy_solver,
                                n_samples=1,
                                sample_steps=config.policy_sampling_steps,
                                condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1), w_cfg=1.0,
                                use_ema=config.policy_use_ema, temperature=config.policy_temperature)
                    # NOTE: Clip action to be within valid range [-1, 1]
                    action = act.clip(-1., 1.).squeeze().cpu().numpy()
            else:
                # inverse dynamic
                normalized_subgoal_obs_for_low = normalizer.normalize(current_subgoal_obs)[-goal_dim:]
                with torch.no_grad():
                    action = low_controller.predict(normalized_current_obs, normalized_subgoal_obs_for_low).clip(-1., 1.).squeeze().cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_done = info['success']
            if episode_done:
                print(f"Episode success after {current_episode_step+1} steps.")
            current_episode_step += 1

            # Rendering for video episodes
            if is_video_episode and (current_episode_step % video_frame_skip == 0 or episode_done):
                frame = env.render().copy()
                if goal_frame_rendered_from_env is not None:
                    rendered_frames_for_this_episode.append(np.concatenate([goal_frame_rendered_from_env, frame], axis=0))
                else:
                    rendered_frames_for_this_episode.append(frame)
            
            # Store transition data
            if not is_video_episode: # Only store full trajectory data for non-video eval episodes
                transition_payload = dict(
                    observation=obs.copy(), next_observation=next_obs.copy(), action=action.copy(),
                    reward=reward, done=bool(terminated or truncated), info=info.copy()
                )
                add_to(current_episode_trajectory_data, transition_payload)
            
            obs = next_obs

        # End of episode
        if not is_video_episode:
            task_trajectories.append(dict(current_episode_trajectory_data))
            # Use the *last* info from the episode for summary stats
            add_to(task_stats_collector, flatten(info))
        else: # Video episode
            if rendered_frames_for_this_episode:
                task_renders_list.append(np.array(rendered_frames_for_this_episode))

    # Aggregate statistics for this task
    aggregated_stats_for_this_task = {}
    for k, v_list in task_stats_collector.items():
        numeric_vals = [x for x in v_list if isinstance(x, (int, float, np.number))]
        if numeric_vals:
            aggregated_stats_for_this_task[k] = np.mean(numeric_vals)
        elif v_list: # Handle non-numeric summary data if any (e.g. list of strings)
            aggregated_stats_for_this_task[k] = v_list

    return aggregated_stats_for_this_task, task_trajectories, task_renders_list, task_trajectories_3d


def low_controller_evaluate(
    low_controller,
    env,
    normalizer,
    task_id,
    obs_dim,
    act_dim,
    config,
    num_eval_episodes,
    num_video_episodes,
    video_frame_skip,
):
    """Evaluate only a low-level controller on a single OGBench task.

    与 single_layer_evaluate 不同，这里不做高层规划，直接让 low_controller
    在 (obs, goal) 条件下选择动作。为了简单起见，我们用环境给出的 overall goal
    作为条件 g：low_controller.predict(obs, goal_state)。
    """

    task_trajectories = []
    task_stats_collector = defaultdict(list)
    task_renders_list = []

    for i_episode in trange(num_eval_episodes + num_video_episodes, desc=f"Task {task_id} Episodes", leave=False):
        current_episode_trajectory_data = defaultdict(list)
        is_video_episode = i_episode >= num_eval_episodes
        reset_options = {"task_id": task_id}
        reset_options["render_goal"] = is_video_episode

        obs, info = env.reset(options=reset_options)
        task_overall_goal_state = info.get("goal")
        goal_frame_rendered_from_env = info.get("goal_rendered")

        episode_done = False
        current_episode_step = 0
        rendered_frames_for_this_episode = []

        while (not episode_done) and (current_episode_step < max_episode_lengths[config.task.env_name]):
            # 直接用环境给的 final goal 作为条件
            normalized_current_obs = normalizer.normalize(obs[:obs_dim][None])
            normalized_task_goal = normalizer.normalize(task_overall_goal_state[:obs_dim][None])

            with torch.no_grad():
                obs_tensor = torch.tensor(normalized_current_obs, device=config.device, dtype=torch.float32).squeeze(0)
                goal_tensor = torch.tensor(normalized_task_goal, device=config.device, dtype=torch.float32).squeeze(0)
                action = (
                    low_controller.predict(obs_tensor, goal_tensor)
                    .clip(-1.0, 1.0)
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_done = info.get("success", bool(terminated or truncated))
            current_episode_step += 1

            if is_video_episode and (current_episode_step % video_frame_skip == 0 or episode_done):
                frame = env.render().copy()
                if goal_frame_rendered_from_env is not None:
                    rendered_frames_for_this_episode.append(
                        np.concatenate([goal_frame_rendered_from_env, frame], axis=0)
                    )
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

    aggregated_stats_for_this_task = {}
    for k, v_list in task_stats_collector.items():
        numeric_vals = [x for x in v_list if isinstance(x, (int, float, np.number))]
        if numeric_vals:
            aggregated_stats_for_this_task[k] = np.mean(numeric_vals)
        elif v_list:
            aggregated_stats_for_this_task[k] = v_list

    return aggregated_stats_for_this_task, task_trajectories, task_renders_list
