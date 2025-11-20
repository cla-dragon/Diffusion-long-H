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
    horizon,
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
    # Templates for planning priors
    prior_template = torch.zeros((1, horizon, obs_dim), device=config.device)

    task_trajectories = []
    task_stats_collector = defaultdict(list)
    task_renders_list = []

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
        current_subgoal_for_gciql = None   
        current_subgoal_idx_in_plan = -1 

        while (not episode_done) and (current_episode_step < max_episode_lengths[config.task.env_name]):
            # if current_episode_step == 0 or \
            #    (config.task.replan_every > 0 and current_episode_step % config.task.replan_every == 0):
            normalized_current_obs = normalizer.normalize(obs[:obs_dim][None])
            normalized_task_goal = normalizer.normalize(task_overall_goal_state[:obs_dim][None])

            # Create priors for planning
            current_prior = prior_template.clone()
            current_prior[:, 0] = torch.tensor(normalized_current_obs, device=config.device, dtype=torch.float32)
            current_prior[:, -1] = torch.tensor(normalized_task_goal, device=config.device, dtype=torch.float32)

            # Planning
            traj_normalized, _ = diffusions_model.sample(
                current_prior.repeat(config.planner_num_candidates, 1, 1),
                solver=config.planner_solver, n_samples=config.planner_num_candidates,
                sample_steps=config.planner_sampling_steps, use_ema=config.planner_use_ema,
                temperature=config.task.planner_temperature,
                w_ldg=config.task.w_ldg
            )
            # Select the best plan
            # TODO: N-step Q value estimator
            idx = 0
            if config.use_diffusion_invdyn:
                policy_prior = torch.zeros((1, act_dim), device=config.device)
                with torch.no_grad():
                    next_obs_plan = traj_normalized[idx, 1, :].unsqueeze(0)
                    obs_policy = torch.tensor(obs.copy(), device=config.device, dtype=torch.float32).unsqueeze(0)
                    next_obs_policy = next_obs_plan.clone()
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
                with torch.no_grad():
                    action = low_controller.predict(obs, traj_normalized[:, 1, :]).clip(-1., 1.).cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_done = info['success']
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

    return aggregated_stats_for_this_task, task_trajectories, task_renders_list, current_plan_observations, current_subgoal_for_gciql
