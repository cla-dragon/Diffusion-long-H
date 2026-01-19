import os
import ogbench
import hydra, wandb, uuid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import trange

import torch
from torch.utils.data import DataLoader
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import set_seed
from cleandiffuser_sup.datasets.ogbench_dataset import OGBenchDataset, GCDataset
from cleandiffuser_sup.lowcontrol.gciql_inv import GCIQLAgent
from evaluate import low_controller_evaluate
from pipelines.utils import get_wandb_video

def detect_subgoals(agent, trajectories, k_goals, c_lookback, device, min_traj_length=20, min_subgoal_dist=30):
    """
    Analyzes trajectories to find subgoals using the Value function.
    
    Args:
        agent: GCIQL agent with trained value_net
        trajectories: List of trajectory arrays (T, obs_dim)
        k_goals: Number of segments to divide trajectory into
        c_lookback: Lookback multiplier (lookback window = c * T/k)
        device: torch device
    
    Returns:
        List of results dicts, one for each trajectory. Each dict contains:
            - 'subgoal_votes': np.array (T,), weighted votes for each timestep being a subgoal
            - 'curves': List of tuples (indices, values) for visualization
            - 'valid_subgoals': List of indices identified as subgoals
    """
    results = []
    
    for traj in trajectories:
        T = len(traj)
        if T < min_traj_length:
            results.append(None)
            continue
            
        L = T / k_goals
        subgoal_votes = np.zeros(T)
        curves = []
        
        # Calculate goal indices (e.g. if k=4, goals at roughly 25%, 50%, 75%, 100%)
        # np.linspace gives k+1 points, slice [1:] to get the k goals
        goal_indices = np.linspace(0, T-1, k_goals+1, dtype=int)[1:]
        
        lookback_steps = int(c_lookback * L)
        
        for g_idx in goal_indices:
            start_idx = max(0, g_idx - lookback_steps)
            indices = np.arange(start_idx, g_idx + 1)
            
            if len(indices) < 5:
                continue
                
            segment_obs = traj[indices]
            goal_obs = traj[g_idx]
            
            # Prepare batch for Value Net
            obs_tensor = torch.from_numpy(segment_obs).to(device).float()
            goal_tensor = torch.from_numpy(goal_obs).to(device).float().unsqueeze(0).expand(len(segment_obs), -1)
            
            with torch.no_grad():
                # Values: (Seq_len,)
                values = agent.value_net(obs_tensor, goal_tensor).cpu().numpy().squeeze()
            
            curves.append((indices, values))
            
            # Subgoal Finding Logic: "Valley" detection
            # 1. Smooth the value curve to ignore small oscillations
            # If sequence is short, reduce window size
            smooth_win = min(5, len(values)//2)
            if smooth_win > 1:
                values_smooth = uniform_filter1d(values, size=smooth_win)
            else:
                values_smooth = values
                
            # 2. Find local minima (valleys)
            # We use find_peaks on negative values
            # Prominence helps ignore noise
            val_range = np.max(values_smooth) - np.min(values_smooth) + 1e-6
            prominence = 0.05 * val_range # Threshold: 5% of range
            
            peaks, _ = find_peaks(-values_smooth, prominence=prominence, distance=3)
            
            # Add votes
            for p in peaks:
                global_idx = indices[p]
                # Simple voting: +1. Can use depth as weight if needed.
                subgoal_votes[global_idx] += 1
        
        # Aggregate votes: perform a final smoothing or peak finding on votes
        # Fix: Use convolution to count votes in a window instead of averaging which reduces height
        # Window size 10 means we aggregate votes within ~10 steps
        vote_density = np.convolve(subgoal_votes, np.ones(10), mode='same')
        
        # Find peaks in vote density with minimum distance constraint
        # height=1 means at least 1 vote in the window
        # distance=min_subgoal_dist ensures separation
        final_peaks, _ = find_peaks(vote_density, height=0.5, distance=min_subgoal_dist)
            
        results.append({
            'subgoal_votes': subgoal_votes,
            'vote_density': vote_density,
            'curves': curves,
            'valid_subgoals': final_peaks,
            'T': T
        })
        
    return results

def draw_trajectory_segments(results, trajectories, object_num_val=1, n_splits=4):
    """
    Visualizes the trajectories of cubes between subgoals in 3D.
    Splits each trajectory into n_splits subplots for better visibility.
    """
    if not results:
        return Image.new('RGB', (100, 100), color='white')
    
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    if not valid_indices:
        return Image.new('RGB', (100, 100), color='white')
    
    n_trajs = len(valid_indices)
    
    # Grid: Rows = Trajectories, Cols = Splits
    # Each row shows one full trajectory broken into n_splits parts
    fig, axes = plt.subplots(n_trajs, n_splits, figsize=(5 * n_splits, 5 * n_trajs), 
                             constrained_layout=True, subplot_kw={'projection': '3d'})
    
    # Handle axes array shape to be consistently (n_trajs, n_splits)
    if n_trajs == 1 and n_splits == 1:
        axes = np.array([[axes]])
    elif n_trajs == 1:
        axes = np.expand_dims(axes, 0) # (1, n_splits)
    elif n_splits == 1:
        # If subplot returns 1D array of axes
        if isinstance(axes, np.ndarray) and axes.ndim == 1:
             axes = np.expand_dims(axes, 1) # (n_trajs, 1)
        else:
             axes = np.array([[axes]]).T
    elif isinstance(axes, np.ndarray) and axes.ndim == 1:
        axes = axes.reshape(n_trajs, n_splits)
        
    for idx, traj_idx in enumerate(valid_indices):
        res = results[traj_idx]
        traj = trajectories[traj_idx] # (T, obs_dim)
        
        # Define all key points: Start -> Subgoal 1 -> ... -> Subgoal N -> Goal
        # key_points contains all time indices that define segment boundaries
        key_points = np.concatenate([[0], res['valid_subgoals'], [len(traj)-1]])
        key_points = np.unique(np.sort(key_points)).astype(int)
        
        # We have len(key_points)-1 segments
        num_segments = len(key_points) - 1
        
        # Determine how many segments to show in each split
        segs_per_split = int(np.ceil(num_segments / n_splits))
        if segs_per_split < 1: segs_per_split = 1
        
        cmap = plt.get_cmap('tab10')
        
        for part_j in range(n_splits):
            ax = axes[idx, part_j]
            
            # Determine segment range for this part: [k_start, k_end)
            k_start = part_j * segs_per_split
            k_end = min((part_j + 1) * segs_per_split, num_segments)
            
            if k_start >= num_segments:
                ax.axis('off')
                continue
            
            # Identify timepoints for this split
            # The split covers the path from key_points[k_start] to key_points[k_end]
            
            used_labels = set()
            
            for obj_i in range(object_num_val):
                start_dim = -9 * (obj_i + 1)
                end_dim = start_dim + 3
                obj_traj_xyz = traj[:, start_dim:end_dim]
                
                # Draw segments
                for k in range(k_start, k_end):
                    t_start = key_points[k]
                    t_end = key_points[k+1] + 1 # Slice inclusive for points
                    
                    segment = obj_traj_xyz[t_start:t_end]
                    
                    # Color based on segment index
                    color = cmap(k % 10)
                    ls = '-' if obj_i == 0 else '--'
                    
                    ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], 
                            color=color, linestyle=ls, linewidth=1.5, alpha=0.8)
                
                # Markers for View Start and View End
                # 1. View Start (Green Circle)
                t_vs = key_points[k_start]
                label_vs = "Start" if (obj_i==0 and "Start" not in used_labels) else ""
                if label_vs: used_labels.add("Start")
                ax.scatter(obj_traj_xyz[t_vs, 0], obj_traj_xyz[t_vs, 1], obj_traj_xyz[t_vs, 2],
                           color='green', marker='o', s=60, label=label_vs)
                
                # 2. View End (Red X)
                t_ve = key_points[k_end]
                label_ve = "End" if (obj_i==0 and "End" not in used_labels) else ""
                if label_ve: used_labels.add("End")
                ax.scatter(obj_traj_xyz[t_ve, 0], obj_traj_xyz[t_ve, 1], obj_traj_xyz[t_ve, 2],
                           color='red', marker='X', s=60, label=label_ve)
                
                # 3. Intermediate Subgoals (Blue Star)
                if k_end > k_start + 1:
                    inter_indices = key_points[k_start+1 : k_end]
                    sg_xyz = obj_traj_xyz[inter_indices]
                    label_sg = "Subgoal" if (obj_i==0 and "Subgoal" not in used_labels) else ""
                    if label_sg: used_labels.add("Subgoal")
                    if len(sg_xyz) > 0:
                        ax.scatter(sg_xyz[:, 0], sg_xyz[:, 1], sg_xyz[:, 2],
                                   color='blue', marker='*', s=80, zorder=5, label=label_sg)
            
            # Titles
            t_start_val = key_points[k_start]
            t_end_val = key_points[k_end]
            ax.set_title(f"T{traj_idx} Part {part_j+1}\nStep: {t_start_val}->{t_end_val}")
            
            # Add legend to every plot to show what Start/End mean
            ax.legend(loc='best', prop={'size': 8})

    # Hide unused axes in valid rows (if any remaining slots in row)
    for idx in range(len(valid_indices)):
         for part_j in range(n_splits):
             # Logic inside loop handles turning off axes if k_start >= num_segments
             pass

    # Hide unused axes for invalid rows (if any)
    for idx in range(n_trajs, axes.shape[0]):
        for part_j in range(n_splits):
            axes[idx, part_j].axis('off')
        
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return image

def draw_subgoal_analysis(results, k_goals):
    """
    Visualizes the subgoal analysis results.
    """
    if not results:
        return Image.new('RGB', (100, 100), color='white')
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return Image.new('RGB', (100, 100), color='white')
        
    n_trajs = len(valid_results)
    
    # Create figure: Rows = trajectories
    # Cols = 2: Left (Value Curves), Right (Subgoal Votes)
    fig, axes = plt.subplots(n_trajs, 2, figsize=(12, 3 * n_trajs), constrained_layout=True)
    if n_trajs == 1:
        axes = np.array([axes]) # Ensure 2D
        
    for i, res in enumerate(valid_results):
        T = res['T']
        
        # Plot 1: Value Curves
        ax_val = axes[i, 0]
        ax_val.set_title(f"Traj {i}: Backward Value Curves (k={k_goals})")
        ax_val.set_xlabel("Timestep")
        ax_val.set_ylabel("Value")
        
        for (indices, values) in res['curves']:
            ax_val.plot(indices, values, alpha=0.7)
            # Mark the query goal (end of segment)
            ax_val.axvline(x=indices[-1], color='gray', linestyle=':', alpha=0.3)
            
        # Plot 2: Subgoal Votes
        ax_vote = axes[i, 1]
        ax_vote.set_title(f"Traj {i}: Aggregate Subgoal Votes")
        ax_vote.set_xlabel("Timestep")
        ax_vote.set_ylabel("Vote Score")
        ax_vote.plot(res['subgoal_votes'], color='lightgray', label='Raw Votes')
        if 'vote_density' in res:
             ax_vote.plot(res['vote_density'], color='black', alpha=0.8, label='Vote Density')
        
        # Mark final subgoals
        for sg_idx in res['valid_subgoals']:
            ax_vote.axvline(x=sg_idx, color='red', linestyle='--', alpha=0.8, label='Subgoal')
            # Also mark on value plot
            ax_val.axvline(x=sg_idx, color='red', linestyle='--', alpha=0.3)
            
        # Avoid duplicate labels
        handles, labels = ax_vote.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax_vote.legend(by_label.values(), by_label.keys())
            
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return image
    


object_num = {
    "cube-single-play-v0": 1,
    "cube-double-play-v0": 2,
    "cube-triple-play-v0": 3,
}

@hydra.main(config_path="../configs/diffuser_test/ogbench", config_name="ogbench", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.run_alias) + "_subgoal_disc_" + str(args.mode),
            config=OmegaConf.to_container(args, resolve=True),
        )

    set_seed(args.seed)

    save_path = f"results/{args.pipeline_name}/{args.task.env_name}_SUBGOAL/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ---------------------- Create Env & Dataset ----------------------
    env, dataset, _ = ogbench.make_env_and_datasets(
        args.task.env_name,
        compact_dataset=True,
    )
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    planner_dataset = OGBenchDataset(
        dataset,
        horizon=args.task.planner_horizon,
        max_path_length=args.task.max_path_length,
    )

    policy_dataset = GCDataset(
        dataset,
        args.low_controller,
        planner_dataset.get_normalizer(),
        preprocess_frame_stack=False,
    )
    policy_dataloader = DataLoader(
        policy_dataset,
        batch_size=args.batch_size, # Use standard batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---------------------- Build Low Controller (GCIQL) ----------------------
    invdyn = GCIQLAgent(
        obs_dim=obs_dim,
        action_dim=act_dim,
        goal_dim=obs_dim,
        config=args.low_controller,
        device=args.device,
    )

    # ---------------------- Training ----------------------
    if args.mode == "train":
        # Parameters for Subgoal Discovery
        analysis_warm_steps = 1000            # Wait until value func is decent
        subgoal_k = 20                        # Divide trajectory into k parts
        subgoal_c = 6.0                       # Lookback multiplier
        num_analysis_trajs = 4                # Number of trajectories to analyze
        
        # Pre-sample trajectories for consistent analysis
        fixed_trajs = policy_dataset.get_fixed_trajectories(num_trajectories=num_analysis_trajs, seed=args.seed)

        invdyn.train()
        n_gradient_step = 0
        log = {
            "gradient_steps": 0,
            "policy_loss_value": 0.0,
            "policy_loss_critic": 0.0,
            "policy_loss_actor": 0.0,
            "q_loss": 0.0,
            "bc_loss": 0.0,
            "bc_log_prob": 0.0,
        }

        from tqdm import tqdm
        pbar = tqdm(total=args.invdyn_gradient_steps / args.log_interval)

        for batch in loop_dataloader(policy_dataloader):
            info = invdyn.update(batch)
            log["policy_loss_value"] += info["value/value_loss"]
            log["policy_loss_critic"] += info["critic/critic_loss"]
            log["policy_loss_actor"] += info["actor/actor_loss"]
            log["q_loss"] += info["actor/q_loss"]
            log["bc_loss"] += info["actor/bc_loss"]
            log["bc_log_prob"] += info["actor/bc_log_prob"]

            n_gradient_step += 1

            if n_gradient_step % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step
                for k in log.keys():
                    if k != "gradient_steps":
                        log[k] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step)
                pbar.update(1)
                log = {
                    "gradient_steps": 0,
                    "policy_loss_value": 0.0,
                    "policy_loss_critic": 0.0,
                    "policy_loss_actor": 0.0,
                    "q_loss": 0.0,
                    "bc_loss": 0.0,
                    "bc_log_prob": 0.0,
                }
            
            # Evaluate & Analyze
            if n_gradient_step % args.eval_interval == 0:
                invdyn.eval()
                eval_metrics = {}
                
                # --- Standard Evaluation ---
                renders = []
                overall_metrics = defaultdict(list)
                task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
                num_tasks = len(task_infos)

                for task_id in trange(1, num_tasks + 1):
                    task_name = task_infos[task_id - 1]["task_name"]
                    eval_info, trajs, cur_renders = low_controller_evaluate(
                        low_controller=invdyn,
                        env=env,
                        normalizer=planner_dataset.get_normalizer(),
                        task_id=task_id,
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        config=args,
                        num_eval_episodes=args.num_eval_episodes,
                        num_video_episodes=args.num_video_episodes,
                        video_frame_skip=args.video_frame_skip,
                    )
                    renders.extend(cur_renders)
                    metric_names = ["success"]
                    eval_metrics.update(
                        {f"evaluation/{task_name}_{k}": v for k, v in eval_info.items() if k in metric_names}
                    )
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)
                
                # Overall standard metrics
                for k, v in overall_metrics.items():
                    eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)
                if args.num_video_episodes > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics['video'] = video 

                # --- New: Subgoal Discovery Analysis ---
                if n_gradient_step > analysis_warm_steps and args.enable_wandb:
                    analysis_results = detect_subgoals(
                        invdyn, 
                        fixed_trajs, 
                        k_goals=subgoal_k, 
                        c_lookback=subgoal_c, 
                        device=args.device,
                        min_subgoal_dist=30 # Enforce min distance
                    )
                    analysis_plot = draw_subgoal_analysis(analysis_results, subgoal_k)
                    eval_metrics['analysis/subgoal_discovery'] = wandb.Image(analysis_plot)
                    
                    obj_num_val = object_num.get(args.task.env_name, 1)
                    traj_plot = draw_trajectory_segments(analysis_results, fixed_trajs, object_num_val=obj_num_val)
                    eval_metrics['analysis/trajectory_segments'] = wandb.Image(traj_plot)
                    
                if args.enable_wandb:
                    wandb.log(eval_metrics, step=n_gradient_step)
                
                invdyn.train()

            if n_gradient_step % args.save_interval == 0:
                invdyn.save(os.path.join(save_path, f"{args.run_alias}_subgoal_ckpt_{n_gradient_step}.pt"))

            if n_gradient_step >= args.invdyn_gradient_steps:
                break
        
    elif args.mode == "inference":
        subgoal_k = 20                        # Divide trajectory into k parts
        subgoal_c = 6.0                       # Lookback multiplier
        num_analysis_trajs = 4                # Number of trajectories to analyze
        
        # Pre-sample trajectories for consistent analysis
        fixed_trajs = policy_dataset.get_fixed_trajectories(num_trajectories=num_analysis_trajs, seed=args.seed)

        # ---------------------- Load Model ----------------------
        model_path = os.path.join(save_path, f'{args.run_alias}_subgoal_ckpt_800000.pt') # args.invdyn_ckpt_path
        invdyn.load(model_path)
        invdyn.eval()
        eval_metrics = {}
        # ---------------------- Subgoal Discovery Analysis ----------------------
        analysis_results = detect_subgoals(
            invdyn, 
            fixed_trajs, 
            k_goals=subgoal_k, 
            c_lookback=subgoal_c, 
            device=args.device,
            min_subgoal_dist=30 # Enforce min distance
        )
        analysis_plot = draw_subgoal_analysis(analysis_results, subgoal_k)
        eval_metrics['analysis/subgoal_discovery'] = wandb.Image(analysis_plot)
        
        obj_num_val = object_num.get(args.task.env_name, 1)
        traj_plot = draw_trajectory_segments(analysis_results, fixed_trajs, object_num_val=obj_num_val)
        eval_metrics['analysis/trajectory_segments'] = wandb.Image(traj_plot)
        
        # Do not evaluate standard metrics in inference mode for now!

        if args.enable_wandb:
            wandb.log(eval_metrics)
    
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented.")

if __name__ == "__main__":
    pipeline()
