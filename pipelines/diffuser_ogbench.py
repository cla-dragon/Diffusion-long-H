import os
import csv
import ogbench
import gymnasium
from ogbench.utils import load_dataset
from collections import defaultdict
import hydra, wandb, uuid, tempfile
import numpy as np
from tqdm import tqdm
from tqdm import trange
from omegaconf import OmegaConf
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser_sup.classifier import GCDistance
from cleandiffuser.diffusion import DiscreteDiffusionSDE, ContinuousDiffusionSDE
from cleandiffuser_sup.diffusion import RepaintContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d, DiT1d, DVInvMlp
from cleandiffuser.nn_classifier import MLPNNClassifier
from cleandiffuser.nn_condition import MLPCondition, IdentityCondition
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser_sup.lowcontrol.gciql_inv import GCIQLAgent
from cleandiffuser.utils import report_parameters, set_seed

from cleandiffuser_sup.datasets.ogbench_dataset import OGBenchDataset, GCDataset
from evaluate import single_layer_evaluate
from pipelines.lowcontroller_ogbench import train_low_controller_if_needed, load_low_controller_checkpoint
from pipelines.utils import get_wandb_video, visualize_3d_trajectories_wandb, resolve_goal_indices

object_num = {
    "cube-single-play-v0": 1,
    "cube-double-play-v0": 2,
    "cube-triple-play-v0": 3,
}


def _stack_3d_trajectories_with_padding(trajectory_list):
    """Stack list of [object_num, T, 3] arrays into [B, object_num, T_max, 3] via repeat-last padding."""
    if len(trajectory_list) == 0:
        return np.zeros((0, 0, 0, 3), dtype=np.float32)

    object_num_val = trajectory_list[0].shape[0]
    max_t = max(traj.shape[1] for traj in trajectory_list)
    out = np.zeros((len(trajectory_list), object_num_val, max_t, 3), dtype=np.float32)

    for i, traj in enumerate(trajectory_list):
        if traj.shape[0] != object_num_val:
            raise ValueError(
                f"Inconsistent object_num in trajectory list: expected {object_num_val}, got {traj.shape[0]} at index {i}."
            )

        t = traj.shape[1]
        out[i, :, :t, :] = traj.astype(np.float32)
        if t < max_t:
            out[i, :, t:, :] = traj[:, -1:, :]

    return out

@hydra.main(config_path="../configs/diffuser_test/ogbench", config_name="ogbench", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.run_alias)+"_"+str(args.mode),
            config=OmegaConf.to_container(args, resolve=True)
        )

    set_seed(args.seed)
    # TODO: change save_path
    save_path = f'results/{args.pipeline_name}/{args.task.env_name}_H{args.task.planner_horizon}/'
    lowctrl_save_path = f"results/{args.pipeline_name}/{args.task.env_name}_LOW/"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    if os.path.exists(lowctrl_save_path) is False:
        os.makedirs(lowctrl_save_path)

    # ---------------------- Create Dataset ----------------------
    env, dataset, _ = ogbench.make_env_and_datasets(
        args.task.env_name,
        compact_dataset=True,
    )
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    goal_indices = resolve_goal_indices(args.task, obs_dim)
    goal_dim = int(goal_indices.size)
    goal_idx_tensor = torch.as_tensor(goal_indices, dtype=torch.long)
    
    jump_steps = args.task.get('jump_steps', 1)
    
    planner_dataset = OGBenchDataset(
        dataset,
        horizon=args.task.planner_horizon,
        max_path_length=args.task.max_path_length,
        jump_steps=jump_steps,
    )
    if args.use_diffusion_invdyn:
        policy_dataset = OGBenchDataset(
            dataset,
            horizon=args.task.invdyn_horizon,
            max_path_length=args.task.max_path_length,
            jump_steps=1,
        )
    else:
        policy_dataset = GCDataset(
            dataset,
            args.low_controller,
            planner_dataset.get_normalizer(),
            preprocess_frame_stack=False,
        )
    planner_dataloader = DataLoader(
        planner_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    policy_dataloader = DataLoader(
        policy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # --------------- Network Architecture -----------------
    nn_diffusion_planner = DiT1d(
            obs_dim, emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model, n_heads=args.planner_d_model//64, depth=args.planner_depth, timestep_emb_type="fourier")
    # nn_classifier = HalfJannerUNet1d(
    #     args.task.horizon, obs_dim*2 + act_dim, out_dim=1, # note: obs_dim*2 for goal-conditioned Q(s,a,g)
    #     model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
    #     timestep_emb_type="positional", kernel_size=3)
    print(f"=============== Parameter Report of Planner ==================================")
    report_parameters(nn_diffusion_planner)
    print(f"==============================================================================")

    # --------------- Classifier Guidance --------------------
    classifier = None
    if args.enable_distance_guidance:
        nn_classifier = MLPNNClassifier(1,1,1,[8,8]) # not used
        distance_dims = []
        for i in range(object_num.get(args.task.env_name, 1)):
            distance_dims.extend(list(range(-9*i-9, -9*i-6)))  # x,y,z of each object
        distance_dims = torch.tensor(distance_dims, device=args.device)
        classifier = GCDistance(nn_classifier, device=args.device, distance_dims=distance_dims)

    # ----------------- Masking -------------------
    model_horizon = (args.task.planner_horizon - 1) // jump_steps + 1
    fix_mask = torch.zeros((model_horizon, obs_dim))
    if not args.adaptive_replan_horizon:
        fix_mask[0, :obs_dim] = 1.
    if (not args.enable_distance_guidance) and (not args.adaptive_replan_horizon):
        fix_mask[-1, :obs_dim] = 1.  # condition on goal state
    loss_weight = torch.ones((model_horizon, obs_dim))
    # loss_weight[0, obs_dim:] = args.planner_next_obs_loss_weight

    # --------------- Diffusion Model --------------------
    if args.adaptive_replan_horizon:
        planner = RepaintContinuousDiffusionSDE(
            nn_diffusion_planner, nn_condition=None,
            fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.planner_ema_rate,
            device=args.device, predict_noise=args.planner_predict_noise, noise_schedule="linear")
    else:
        planner = ContinuousDiffusionSDE(
            nn_diffusion_planner, nn_condition=None,
            fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.planner_ema_rate,
            device=args.device, predict_noise=args.planner_predict_noise, noise_schedule="linear")
    
    # ---------------------- Inverse Dynamic (Policy) -----------------------
    if args.use_diffusion_invdyn:
        nn_diffusion_invdyn = DVInvMlp(obs_dim, act_dim, emb_dim=64, hidden_dim=args.policy_hidden_dim, timestep_emb_type="positional").to(args.device)
        nn_condition_invdyn = IdentityCondition(dropout=0.0).to(args.device)
        print(f"=============== Parameter Report of Policy ===================================")
        report_parameters(nn_diffusion_invdyn)
        print(f"==============================================================================")
        # --------------- Diffusion Model Actor --------------------
        policy = DiscreteDiffusionSDE(
            nn_diffusion_invdyn, nn_condition_invdyn, predict_noise=args.policy_predict_noise, optim_params={"lr": args.policy_learning_rate},
            x_max=+1. * torch.ones((1, act_dim), device=args.device),
            x_min=-1. * torch.ones((1, act_dim), device=args.device),
            diffusion_steps=args.policy_diffusion_steps, ema_rate=args.policy_ema_rate, device=args.device)
    else:
        invdyn = GCIQLAgent(
            obs_dim=obs_dim,
            action_dim=act_dim,
            goal_dim=goal_dim,
            config=args.low_controller,
            device=args.device,
        )
        print(f"[LowCtrl] goal_dim={goal_dim}, goal_indices={goal_indices.tolist()} (from task.goal_dim)")
    
    # ---------------------- Training ----------------------
    if args.mode == "train":
        planner_lr_scheduler = CosineAnnealingLR(planner.optimizer, args.planner_diffusion_gradient_steps)
        planner.train()
        # Policy
        if args.use_diffusion_invdyn:
            train_low_controller_if_needed(
                args=args,
                low_controller=policy,
                policy_dataloader=policy_dataloader,
                lowctrl_save_path=lowctrl_save_path,
                goal_idx_tensor=None,
                use_diffusion_invdyn=True,
                log_prefix="warmstart_lowctrl",
            )
            policy.eval()
        else:
            # invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)
            #NOTE GCIQL 不考虑 lr_scheduler
            train_low_controller_if_needed(
                args=args,
                low_controller=invdyn,
                policy_dataloader=policy_dataloader,
                lowctrl_save_path=lowctrl_save_path,
                goal_idx_tensor=goal_idx_tensor,
                use_diffusion_invdyn=False,
                log_prefix="warmstart_lowctrl",
            )
            invdyn.eval()

        n_gradient_step = 0
        log = {"gradient_steps": 0, "avg_loss_planner": 0.}
        pbar = tqdm(total=max(1, args.planner_diffusion_gradient_steps // args.log_interval))

        for planner_batch in loop_dataloader(planner_dataloader):

            planner_horizon_obs = planner_batch["obs"]["state"].to(args.device)
            planner_horizon_action = planner_batch["act"].to(args.device)
            planner_horizon_data = planner_horizon_obs

            # ----------- Gradient Step ------------
            log["avg_loss_planner"] += planner.update(planner_horizon_data)['loss']
            planner_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                for key in log.keys():
                    if key != "gradient_steps":
                        log[key] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                pbar.update(1)
                log = {"gradient_steps": 0, "avg_loss_planner": 0.}
                    
            # ----------- Evalutation ------------
            if (n_gradient_step) % args.eval_interval == 0:
                planner.eval()
                if args.use_diffusion_invdyn:
                    policy.eval()
                else:
                    invdyn.eval()

                renders = []
                eval_metrics = {}
                overall_metrics = defaultdict(list)
                overall_trajectories_3d = []
                overall_actual_trajectories_3d = []
                task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
                num_tasks = len(task_infos)
                for task_id in trange(1, num_tasks + 1):
                    task_name = task_infos[task_id - 1]['task_name']
                    eval_info, trajs, cur_renders, trajs_planned_3d, trajs_actual_3d = single_layer_evaluate(
                        diffusions_model=planner,
                        mode=args.low_controller_mode,
                        low_controller=policy if args.use_diffusion_invdyn else invdyn,
                        env=env,
                        normalizer=planner_dataset.get_normalizer(),
                        task_id=task_id,
                        horizon=model_horizon,
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        config=args,
                        num_eval_episodes=args.num_eval_episodes,
                        num_video_episodes=args.num_video_episodes,
                        video_frame_skip=args.video_frame_skip,
                    )
                    renders.extend(cur_renders)
                    if args.num_video_episodes > 0 and len(trajs_planned_3d) > 0 and len(trajs_actual_3d) > 0:
                        overall_trajectories_3d.append(trajs_planned_3d[0]) # only log the first vision episode of each task
                        overall_actual_trajectories_3d.append(trajs_actual_3d[0]) # only log the first vision episode of each task
                    metric_names = ['success']
                    eval_metrics.update(
                        {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                    )
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)

                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

                if args.num_video_episodes > 0 and len(overall_trajectories_3d) > 0 and len(overall_actual_trajectories_3d) > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics['video'] = video
                    # 3D Trajectory Visualization
                    trajs_planned_3d = _stack_3d_trajectories_with_padding(overall_trajectories_3d)
                    trajs_actual_3d = _stack_3d_trajectories_with_padding(overall_actual_trajectories_3d)
                    trajs_image = visualize_3d_trajectories_wandb(
                        planned_traj=trajs_planned_3d,
                        actual_traj=trajs_actual_3d,
                        n_cols=num_tasks,
                    )
                    eval_metrics['3d_trajectories'] = trajs_image

                wandb.log(eval_metrics, step=n_gradient_step + 1)

                planner.train()
            
            # ----------- Save Model ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                planner.save(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_{n_gradient_step + 1}.pt"))
                planner.save(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_latest.pt"))


            n_gradient_step += 1
            if n_gradient_step >= args.planner_diffusion_gradient_steps:
                print(f"===================== Training Finished =====================")
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        planner.load(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_{args.planner_ckpt}.pt"))
        planner.eval()
        if args.use_diffusion_invdyn:
            load_low_controller_checkpoint(
                args=args,
                low_controller=policy,
                lowctrl_save_path=lowctrl_save_path,
                use_diffusion_invdyn=True,
            )
            policy.eval()
        else:
            load_low_controller_checkpoint(
                args=args,
                low_controller=invdyn,
                lowctrl_save_path=lowctrl_save_path,
                use_diffusion_invdyn=False,
            )
            invdyn.eval()

        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        overall_trajectories_3d = []
        overall_actual_trajectories_3d = []
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        num_tasks = len(task_infos)
        for task_id in trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]['task_name']
            eval_info, trajs, cur_renders, trajs_planned_3d, trajs_actual_3d = single_layer_evaluate(
                diffusions_model=planner,
                mode=args.low_controller_mode,
                low_controller=policy if args.use_diffusion_invdyn else invdyn,
                env=env,
                normalizer=planner_dataset.get_normalizer(),
                task_id=task_id,
                horizon=model_horizon,
                obs_dim=obs_dim,
                act_dim=act_dim,
                config=args,
                num_eval_episodes=args.num_eval_episodes,
                num_video_episodes=args.num_video_episodes,
                video_frame_skip=args.video_frame_skip,
            )
            renders.extend(cur_renders)
            if args.num_video_episodes > 0 and len(trajs_planned_3d) > 0 and len(trajs_actual_3d) > 0:
                overall_trajectories_3d.append(trajs_planned_3d[0]) # only log the first vision episode of each task
                overall_actual_trajectories_3d.append(trajs_actual_3d[0]) # only log the first vision episode of each task
            metric_names = ['success']
            eval_metrics.update(
                {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
            )
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

        for k, v in overall_metrics.items():
            eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

        if args.num_video_episodes > 0 and len(overall_trajectories_3d) > 0 and len(overall_actual_trajectories_3d) > 0:
            video = get_wandb_video(renders=renders, n_cols=num_tasks)
            eval_metrics['video'] = video
            # 3D Trajectory Visualization
            trajs_planned_3d = _stack_3d_trajectories_with_padding(overall_trajectories_3d)
            trajs_actual_3d = _stack_3d_trajectories_with_padding(overall_actual_trajectories_3d)
            trajs_image = visualize_3d_trajectories_wandb(
                planned_traj=trajs_planned_3d,
                actual_traj=trajs_actual_3d,
                n_cols=num_tasks,
            )
            eval_metrics['3d_trajectories'] = trajs_image

        wandb.log(eval_metrics, step=1)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    

if __name__ == "__main__":
    pipeline()