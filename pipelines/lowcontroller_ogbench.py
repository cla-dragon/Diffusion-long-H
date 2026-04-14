import os
import ogbench
import hydra, wandb, uuid
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import trange, tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import set_seed
from cleandiffuser_sup.datasets.ogbench_dataset import OGBenchDataset, GCDataset
from cleandiffuser_sup.lowcontrol.gciql_inv import GCIQLAgent
from evaluate import low_controller_evaluate
from pipelines.utils import get_wandb_video, resolve_goal_indices


def _cfg_get(cfg, key, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _lowctrl_alias(args):
    return str(_cfg_get(args, "lowctrl_alias", _cfg_get(args, "run_alias", "lowctrl")))


def load_low_controller_checkpoint(args, low_controller, lowctrl_save_path, use_diffusion_invdyn=False, ckpt=None):
    alias = _lowctrl_alias(args)
    ckpt_name = str(ckpt) if ckpt is not None else str(_cfg_get(args, "policy_ckpt" if use_diffusion_invdyn else "invdyn_ckpt", "latest"))
    load_path = os.path.join(lowctrl_save_path, f"{alias}_lowctrl_ckpt_{ckpt_name}.pt")
    low_controller.load(load_path)
    return load_path


def train_low_controller_if_needed(
    args,
    low_controller,
    policy_dataloader,
    lowctrl_save_path,
    goal_idx_tensor=None,
    use_diffusion_invdyn=False,
    log_prefix="lowctrl",
    **_unused_kwargs,
):
    alias = _lowctrl_alias(args)
    lowctrl_type = "diffusion_invdyn" if use_diffusion_invdyn else "gciql"
    default_ckpt_path = os.path.join(lowctrl_save_path, f"{lowctrl_type}_{alias}_lowctrl_ckpt_latest.pt")
    load_existing_alias = bool(
        _cfg_get(args, "lowctrl_load_existing_alias", _cfg_get(args, "lowctrl_load_existing", False))
    )

    if load_existing_alias:
        if not os.path.exists(default_ckpt_path):
            raise FileNotFoundError(
                f"[LowCtrl] lowctrl_load_existing_alias=True, but checkpoint not found: {default_ckpt_path}"
            )
        print(f"[LowCtrl] Load existing alias checkpoint: {default_ckpt_path}")
        low_controller.load(default_ckpt_path)
        return {"status": "loaded", "path": default_ckpt_path}

    target_steps = int(
        _cfg_get(args, "policy_diffusion_gradient_steps" if use_diffusion_invdyn else "invdyn_gradient_steps", 0)
    )
    if target_steps <= 0:
        raise ValueError("[LowCtrl] target_steps must be > 0 when lowctrl_load_existing_alias=False")

    print(f"[LowCtrl] Start training ({'diffusion_invdyn' if use_diffusion_invdyn else 'gciql'}) for {target_steps} steps")
    low_controller.train()
    scheduler = None
    if use_diffusion_invdyn:
        scheduler = CosineAnnealingLR(low_controller.optimizer, target_steps)

    log_interval = int(_cfg_get(args, "log_interval", 1000))
    save_interval = int(_cfg_get(args, "save_interval", 100000))
    log = {"gradient_steps": 0}
    if use_diffusion_invdyn:
        log["bc_loss_policy"] = 0.0
    else:
        log["policy_loss_value"] = 0.0
        log["policy_loss_critic"] = 0.0
        log["policy_loss_actor"] = 0.0
        log["q_loss"] = 0.0
        log["bc_loss"] = 0.0
        log["bc_log_prob"] = 0.0

    pbar = tqdm(total=max(1, target_steps // max(log_interval, 1)), desc="LowCtrl")
    invdyn_horizon = max(2, int(_cfg_get(_cfg_get(args, "task", {}), "invdyn_horizon", 2)))

    for n_gradient_step, batch in enumerate(loop_dataloader(policy_dataloader), start=1):
        if use_diffusion_invdyn:
            policy_horizon_obs = batch["obs"]["state"].to(args.device)
            policy_horizon_action = batch["act"].to(args.device)
            invdyn_pick_index = torch.randint(1, invdyn_horizon, (1,)).item()
            policy_td_obs = policy_horizon_obs[:, 0, :]
            policy_td_next_obs = policy_horizon_obs[:, invdyn_pick_index, :]
            policy_td_act = policy_horizon_action[:, 0, :]

            policy_loss = low_controller.update(
                policy_td_act,
                torch.cat([policy_td_obs, policy_td_next_obs], dim=-1),
            )["loss"]
            log["bc_loss_policy"] += float(policy_loss)
            if scheduler is not None:
                scheduler.step()
        else:
            if goal_idx_tensor is not None:
                idx = goal_idx_tensor.to(batch["value_goals"].device)
                batch["value_goals"] = batch["value_goals"].index_select(-1, idx)
                batch["actor_goals"] = batch["actor_goals"].index_select(-1, idx)

            info = low_controller.update(batch)
            log["policy_loss_value"] += float(info.get("value/value_loss", 0.0))
            log["policy_loss_critic"] += float(info.get("critic/critic_loss", 0.0))
            log["policy_loss_actor"] += float(info.get("actor/actor_loss", 0.0))
            log["q_loss"] += float(info.get("actor/q_loss", 0.0))
            log["bc_loss"] += float(info.get("actor/bc_loss", 0.0))
            log["bc_log_prob"] += float(info.get("actor/bc_log_prob", 0.0))

        if n_gradient_step % log_interval == 0:
            out = {"gradient_steps": n_gradient_step}
            out.update({k: v / log_interval for k, v in log.items() if k != "gradient_steps"})
            print(out)
            if bool(_cfg_get(args, "enable_wandb", False)):
                wandb.log({f"{log_prefix}/{k}": v for k, v in out.items()}, step=n_gradient_step)
            pbar.update(1)

            if use_diffusion_invdyn:
                log = {"gradient_steps": 0, "bc_loss_policy": 0.0}
            else:
                log = {
                    "gradient_steps": 0,
                    "policy_loss_value": 0.0,
                    "policy_loss_critic": 0.0,
                    "policy_loss_actor": 0.0,
                    "q_loss": 0.0,
                    "bc_loss": 0.0,
                    "bc_log_prob": 0.0,
                }

        if n_gradient_step % save_interval == 0:
            low_controller.save(os.path.join(lowctrl_save_path, f"{lowctrl_type}_{alias}_lowctrl_ckpt_{n_gradient_step}.pt"))
            low_controller.save(default_ckpt_path)

        if n_gradient_step >= target_steps:
            break

    low_controller.save(default_ckpt_path)
    print(f"[LowCtrl] Training done. Saved: {default_ckpt_path}")
    return {"status": "trained", "path": default_ckpt_path, "steps": target_steps}


def draw_value_curves(agent, trajectories, k_goals, device, goal_indices=None):
    # trajectories: list of np.ndarray (T, obs_dim)
    # k_goals: int
    
    if not trajectories:
        return Image.new('RGB', (100, 100), color='white')

    fig, axes = plt.subplots(1, k_goals, figsize=(5 * k_goals, 4))
    if k_goals == 1:
        axes = [axes]
        
    # For each goal index j (0 to k-1)
    for j in range(k_goals):
        ax = axes[j]
        ax.set_title(f"Goal {j+1}/{k_goals}")
        
        for i, traj in enumerate(trajectories):
            # traj: (T, obs_dim)
            T = len(traj)
            if T < 2:
                continue
                
            # Determine goal index
            # np.linspace(0, T-1, k+1)[1:] gives k points.
            temporal_goal_indices = np.linspace(0, T-1, k_goals+1, dtype=int)[1:]
            g_idx = temporal_goal_indices[j]
            
            goal = traj[g_idx] # (obs_dim,)
            if goal_indices is not None:
                goal = goal[goal_indices]
            
            # Prepare inputs for value net
            obs_tensor = torch.from_numpy(traj).to(device).float()
            goal_tensor = torch.from_numpy(goal).to(device).float().unsqueeze(0).expand(T, -1)
            
            with torch.no_grad():
                values = agent.value_net(obs_tensor, goal_tensor).cpu().numpy().squeeze()
            ax.plot(values, label=f"Traj {i}")
            # Mark the goal position
            ax.axvline(x=g_idx, linestyle="--", alpha=0.5)
            
        if j == 0:
            ax.legend()
            
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return image


@hydra.main(config_path="../configs/diffuser_test/ogbench", config_name="ogbench", version_base=None)
def pipeline(args):
    """Evaluate only the low-level controller on OGBench tasks.

    This ignores the high-level diffusion planner and directly lets
    the low_controller act using (obs, goal) pairs.
    """

    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.run_alias) + "_lowctrl_" + str(args.mode),
            config=OmegaConf.to_container(args, resolve=True),
        )

    set_seed(args.seed)

    save_path = f"results/{args.pipeline_name}/{args.task.env_name}_LOW/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ---------------------- Create Env & Dataset ----------------------
    env, dataset, _ = ogbench.make_env_and_datasets(
        args.task.env_name,
        compact_dataset=True,
    )
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # For evaluation, we still need a normalizer, so reuse OGBenchDataset
    planner_dataset = OGBenchDataset(
        dataset,
        horizon=args.task.planner_horizon,
        max_path_length=args.task.max_path_length,
    )

    # Policy dataset for GCIQL training (goal-conditioned)
    policy_dataset = GCDataset(
        dataset,
        args.low_controller,
        planner_dataset.get_normalizer(),
        preprocess_frame_stack=False,
    )
    policy_dataloader = DataLoader(
        policy_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---------------------- Build Low Controller (GCIQL) ----------------------
    goal_indices = resolve_goal_indices(args.task, obs_dim)
    goal_dim = int(goal_indices.size)
    goal_idx_tensor = torch.as_tensor(goal_indices, dtype=torch.long)

    invdyn = GCIQLAgent(
        obs_dim=obs_dim,
        action_dim=act_dim,
        goal_dim=goal_dim,
        config=args.low_controller,
        device=args.device,
    )

    # ---------------------- Training ----------------------
    if args.mode == "train":
        # Sample trajectories for visualization
        eval_trajectories = policy_dataset.get_fixed_trajectories(num_trajectories=3, seed=args.seed)
        k_goals = 4
        train_low_controller_if_needed(
            args=args,
            low_controller=invdyn,
            policy_dataloader=policy_dataloader,
            lowctrl_save_path=save_path,
            goal_idx_tensor=goal_idx_tensor,
            use_diffusion_invdyn=False,
            log_prefix="lowctrl",
        )

        invdyn.eval()
        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
        num_tasks = len(task_infos)

        for task_id in trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]["task_name"]
            eval_info, _, cur_renders = low_controller_evaluate(
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
                goal_indices=goal_indices,
            )

            renders.extend(cur_renders)
            metric_names = ["success"]
            eval_metrics.update(
                {f"evaluation/{task_name}_{k}": v for k, v in eval_info.items() if k in metric_names}
            )
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

        for k, v in overall_metrics.items():
            eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

        if args.num_video_episodes > 0:
            eval_metrics["video"] = get_wandb_video(renders=renders, n_cols=num_tasks)

        if args.enable_wandb:
            value_curves_plot = draw_value_curves(
                invdyn,
                eval_trajectories,
                k_goals,
                args.device,
                goal_indices=goal_indices,
            )
            eval_metrics["evaluation/value_curves"] = wandb.Image(value_curves_plot)
            wandb.log(eval_metrics, step=int(_cfg_get(args, "invdyn_gradient_steps", 0)))
        
    # ---------------------- Inference / Evaluation ----------------------
    elif args.mode == "inference":
        load_low_controller_checkpoint(
            args=args,
            low_controller=invdyn,
            lowctrl_save_path=save_path,
            use_diffusion_invdyn=False,
        )
        invdyn.eval()

        renders = []
        eval_metrics = {}
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
                goal_indices=goal_indices,
            )

            renders.extend(cur_renders)
            metric_names = ["success"]
            eval_metrics.update(
                {f"evaluation/{task_name}_{k}": v for k, v in eval_info.items() if k in metric_names}
            )
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

        for k, v in overall_metrics.items():
            eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

        if args.num_video_episodes > 0:
            video = get_wandb_video(renders=renders, n_cols=num_tasks)
            eval_metrics['video'] = video 

        if args.enable_wandb:
            wandb.log(eval_metrics, step=1)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
