import os
import uuid
from collections import defaultdict

import hydra
import numpy as np
import ogbench
import torch
import wandb
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.utils import report_parameters, set_seed
from cleandiffuser_sup.datasets.ogbench_dataset import OGBenchDataset, GCDataset
from cleandiffuser_sup.diffusion import MultiSegmentRepaintDiffusionSDE
from cleandiffuser_sup.lowcontrol.gciql_inv import GCIQLAgent
from evaluate import multi_segment_evaluate
from pipelines.lowcontroller_ogbench import train_low_controller_if_needed, load_low_controller_checkpoint
from pipelines.utils import get_wandb_video, resolve_goal_indices, visualize_3d_trajectories_wandb


object_num = {
    "cube-single-play-v0": 1,
    "cube-double-play-v0": 2,
    "cube-triple-play-v0": 3,
}


def _stack_3d_trajectories_with_padding(trajectory_list):
    if len(trajectory_list) == 0:
        return np.zeros((0, 0, 0, 3), dtype=np.float32)

    object_num_val = trajectory_list[0].shape[0]
    max_t = max(traj.shape[1] for traj in trajectory_list)
    out = np.zeros((len(trajectory_list), object_num_val, max_t, 3), dtype=np.float32)

    for i, traj in enumerate(trajectory_list):
        t = traj.shape[1]
        out[i, :, :t, :] = traj.astype(np.float32)
        if t < max_t:
            out[i, :, t:, :] = traj[:, -1:, :]

    return out


def _stack_segment_3d_trajectories_with_padding(segment_trajectory_list):
    if len(segment_trajectory_list) == 0:
        return np.zeros((0, 0, 0, 0, 3), dtype=np.float32)

    max_seg = max(traj.shape[0] for traj in segment_trajectory_list)
    object_num_val = segment_trajectory_list[0].shape[1]
    max_t = max(traj.shape[2] for traj in segment_trajectory_list)
    out = np.zeros((len(segment_trajectory_list), max_seg, object_num_val, max_t, 3), dtype=np.float32)

    for i, traj in enumerate(segment_trajectory_list):
        seg_n, _, t, _ = traj.shape
        out[i, :seg_n, :, :t, :] = traj.astype(np.float32)
        if t < max_t:
            out[i, :seg_n, :, t:, :] = traj[:, :, -1:, :]
        if seg_n < max_seg:
            out[i, seg_n:, :, :, :] = out[i, seg_n - 1 : seg_n, :, :, :]

    return out


@hydra.main(config_path="../configs/diffuser_test/ogbench", config_name="ogbench_multiseg", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"

    if args.enable_wandb and args.mode in ["train", "inference"]:
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.run_alias) + "_" + str(args.mode),
            config=OmegaConf.to_container(args, resolve=True),
        )

    set_seed(args.seed)

    planner_horizon_cfg = int(args.task.get("multiseg_horizon", args.task.planner_horizon))

    save_path = f"results/{args.pipeline_name}/{args.task.env_name}_Multi_SegH{planner_horizon_cfg}/"
    lowctrl_save_path = f"results/{args.pipeline_name}/{args.task.env_name}_LOW/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(lowctrl_save_path, exist_ok=True)

    env, dataset, _ = ogbench.make_env_and_datasets(args.task.env_name, compact_dataset=True)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    goal_indices = resolve_goal_indices(args.task, obs_dim)
    goal_dim = int(goal_indices.size)
    goal_idx_tensor = torch.as_tensor(goal_indices, dtype=torch.long)

    jump_steps = args.task.get("jump_steps", 1)
    planner_dataset = OGBenchDataset(
        dataset,
        horizon=planner_horizon_cfg,
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
        planner_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    policy_dataloader = DataLoader(
        policy_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    nn_diffusion_planner = DiT1d(
        obs_dim,
        emb_dim=args.planner_emb_dim,
        d_model=args.planner_d_model,
        n_heads=args.planner_d_model // 64,
        depth=args.planner_depth,
        timestep_emb_type="fourier",
    )

    print("=============== Parameter Report of Planner ==================================")
    report_parameters(nn_diffusion_planner)
    print("==============================================================================")

    model_horizon = (planner_horizon_cfg - 1) // jump_steps + 1
    fix_mask = torch.zeros((model_horizon, obs_dim))
    loss_weight = torch.ones((model_horizon, obs_dim))

    planner = MultiSegmentRepaintDiffusionSDE(
        nn_diffusion_planner,
        nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        ema_rate=args.planner_ema_rate,
        device=args.device,
        predict_noise=args.planner_predict_noise,
        noise_schedule="linear",
    )

    if args.use_diffusion_invdyn:
        nn_diffusion_invdyn = DVInvMlp(
            obs_dim,
            act_dim,
            emb_dim=64,
            hidden_dim=args.policy_hidden_dim,
            timestep_emb_type="positional",
        ).to(args.device)
        nn_condition_invdyn = IdentityCondition(dropout=0.0).to(args.device)
        policy = DiscreteDiffusionSDE(
            nn_diffusion_invdyn,
            nn_condition_invdyn,
            predict_noise=args.policy_predict_noise,
            optim_params={"lr": args.policy_learning_rate},
            x_max=+1.0 * torch.ones((1, act_dim), device=args.device),
            x_min=-1.0 * torch.ones((1, act_dim), device=args.device),
            diffusion_steps=args.policy_diffusion_steps,
            ema_rate=args.policy_ema_rate,
            device=args.device,
        )
    else:
        invdyn = GCIQLAgent(
            obs_dim=obs_dim,
            action_dim=act_dim,
            goal_dim=goal_dim,
            config=args.low_controller,
            device=args.device,
        )

    if args.mode == "train":
        planner_lr_scheduler = CosineAnnealingLR(planner.optimizer, args.planner_diffusion_gradient_steps)
        planner.train()

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
            log = {"gradient_steps": 0, "avg_loss_planner": 0.0}
        else:
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
            log = {
                "gradient_steps": 0,
                "avg_loss_planner": 0.0,
            }

        pbar = tqdm(total=max(1, args.planner_diffusion_gradient_steps // args.log_interval))

        n_gradient_step = 0
        for planner_batch in loop_dataloader(planner_dataloader):
            planner_horizon_data = planner_batch["obs"]["state"].to(args.device)

            planner_log = planner.update(
                planner_horizon_data,
                overlap_length=int(args.multi_segment.overlap_length),
                overlap_prob=float(args.training_conditioning.overlap_prob),
                inpaint_start_prob=float(args.training_conditioning.inpaint_start_prob),
                inpaint_end_prob=float(args.training_conditioning.inpaint_end_prob),
            )
            log["avg_loss_planner"] += planner_log["loss"]
            planner_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                out = {"gradient_steps": n_gradient_step + 1}
                for k, v in log.items():
                    if k != "gradient_steps":
                        out[k] = v / args.log_interval
                print(out)
                if args.enable_wandb:
                    wandb.log(out, step=n_gradient_step + 1)
                pbar.update(1)

                log = {"gradient_steps": 0, "avg_loss_planner": 0.0}

            if n_gradient_step % args.eval_interval == 0:
                planner.eval()
                if args.use_diffusion_invdyn:
                    policy.eval()
                else:
                    invdyn.eval()

                renders = []
                eval_metrics = {}
                overall_metrics = defaultdict(list)
                overall_trajectories_3d = []
                overall_segment_trajectories_3d = []
                overall_actual_trajectories_3d = []

                task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
                num_tasks = len(task_infos)
                for task_id in trange(1, num_tasks + 1):
                    task_name = task_infos[task_id - 1]["task_name"]
                    eval_info, _, cur_renders, trajs_planned_3d, trajs_planned_segments_3d, trajs_actual_3d = multi_segment_evaluate(
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
                        overall_trajectories_3d.append(trajs_planned_3d[0])
                        if len(trajs_planned_segments_3d) > 0:
                            overall_segment_trajectories_3d.append(trajs_planned_segments_3d[0])
                        overall_actual_trajectories_3d.append(trajs_actual_3d[0])
                    eval_metrics.update({f"evaluation/{task_name}_success": eval_info.get("success", 0.0)})
                    overall_metrics["success"].append(eval_info.get("success", 0.0))

                eval_metrics["evaluation/overall_success"] = np.mean(overall_metrics["success"])

                if args.num_video_episodes > 0 and len(overall_trajectories_3d) > 0 and len(overall_actual_trajectories_3d) > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics["video"] = video
                    trajs_planned_3d = _stack_3d_trajectories_with_padding(overall_trajectories_3d)
                    trajs_planned_segments_3d = _stack_segment_3d_trajectories_with_padding(overall_segment_trajectories_3d)
                    trajs_actual_3d = _stack_3d_trajectories_with_padding(overall_actual_trajectories_3d)
                    trajs_image = visualize_3d_trajectories_wandb(
                        planned_traj=trajs_planned_3d,
                        planned_segment_traj=trajs_planned_segments_3d,
                        actual_traj=trajs_actual_3d,
                        n_cols=num_tasks,
                    )
                    eval_metrics["3d_trajectories"] = trajs_image

                if args.enable_wandb:
                    wandb.log(eval_metrics, step=n_gradient_step + 1)

                planner.train()

            if (n_gradient_step + 1) % args.save_interval == 0:
                planner.save(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_{n_gradient_step + 1}.pt"))
                planner.save(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_latest.pt"))

            n_gradient_step += 1
            if n_gradient_step >= args.planner_diffusion_gradient_steps:
                break

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
        overall_segment_trajectories_3d = []
        overall_actual_trajectories_3d = []

        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
        num_tasks = len(task_infos)
        for task_id in trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]["task_name"]
            eval_info, _, cur_renders, trajs_planned_3d, trajs_planned_segments_3d, trajs_actual_3d = multi_segment_evaluate(
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
                overall_trajectories_3d.append(trajs_planned_3d[0])
                if len(trajs_planned_segments_3d) > 0:
                    overall_segment_trajectories_3d.append(trajs_planned_segments_3d[0])
                overall_actual_trajectories_3d.append(trajs_actual_3d[0])
            eval_metrics[f"evaluation/{task_name}_success"] = eval_info.get("success", 0.0)
            overall_metrics["success"].append(eval_info.get("success", 0.0))

        eval_metrics["evaluation/overall_success"] = np.mean(overall_metrics["success"])

        if args.num_video_episodes > 0 and len(overall_trajectories_3d) > 0 and len(overall_actual_trajectories_3d) > 0:
            video = get_wandb_video(renders=renders, n_cols=num_tasks)
            eval_metrics["video"] = video
            trajs_planned_3d = _stack_3d_trajectories_with_padding(overall_trajectories_3d)
            trajs_planned_segments_3d = _stack_segment_3d_trajectories_with_padding(overall_segment_trajectories_3d)
            trajs_actual_3d = _stack_3d_trajectories_with_padding(overall_actual_trajectories_3d)
            trajs_image = visualize_3d_trajectories_wandb(
                planned_traj=trajs_planned_3d,
                planned_segment_traj=trajs_planned_segments_3d,
                actual_traj=trajs_actual_3d,
                n_cols=num_tasks,
            )
            eval_metrics["3d_trajectories"] = trajs_image

        if args.enable_wandb:
            wandb.log(eval_metrics, step=1)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
