import os
import uuid
from collections import defaultdict
from datetime import datetime

import hydra
import numpy as np
import ogbench
import torch
import wandb
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm, trange

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_classifier import MLPNNClassifier
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.utils import report_parameters, set_seed
from cleandiffuser_sup.classifier import GCDistance
from cleandiffuser_sup.datasets.ogbench_dataset import GCDataset, OGBenchDataset
from cleandiffuser_sup.diffusion import SegmentStitchDiscreteDiffusionSDE
from cleandiffuser_sup.lowcontrol.gciql_inv import GCIQLAgent
from cleandiffuser_sup.nn_condition import SegmentBoundaryCondition
from evaluate import segment_stitch_evaluate
from pipelines.lowcontroller_ogbench import load_low_controller_checkpoint, train_low_controller_if_needed
from pipelines.subgoal_constrained_diffuser_ogbench import (
    _get_trajectory_obs_list,
    _save_histogram_png_and_json,
    _save_subgoal_analysis_summary_json,
    _to_wandb_histogram,
    build_soft_sampling_weights,
    build_subgoal_analysis_tables,
    build_subgoal_filtered_indices,
    detect_subgoals_for_all_trajectories,
)
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

    for idx, traj in enumerate(trajectory_list):
        t = traj.shape[1]
        out[idx, :, :t, :] = traj.astype(np.float32)
        if t < max_t:
            out[idx, :, t:, :] = traj[:, -1:, :]

    return out


def _stack_segment_3d_trajectories_with_padding(trajectory_list):
    if len(trajectory_list) == 0:
        return np.zeros((0, 0, 0, 0, 3), dtype=np.float32)

    max_segments = max(traj.shape[0] for traj in trajectory_list)
    object_num_val = trajectory_list[0].shape[1]
    max_t = max(traj.shape[2] for traj in trajectory_list)
    out = np.zeros((len(trajectory_list), max_segments, object_num_val, max_t, 3), dtype=np.float32)

    for idx, traj in enumerate(trajectory_list):
        num_segments, _, t = traj.shape[:3]
        out[idx, :num_segments, :, :t, :] = traj.astype(np.float32)
        if t < max_t:
            out[idx, :num_segments, :, t:, :] = traj[:, :, -1:, :]
        if num_segments < max_segments:
            out[idx, num_segments:, :, :, :] = traj[-1:, :, -1:, :]

    return out


def _subgoal_filter_enabled(args):
    return bool(args.subgoal_filter.enable)


def _planner_ckpt_stem(args):
    if _subgoal_filter_enabled(args):
        return f"{args.run_alias}_subgoal_constrained_planner"
    return f"{args.run_alias}_planner"


def _build_subgoal_analysis_log_dir(save_path):
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(save_path, "subgoal_filter_logs", run_time)


@hydra.main(config_path="../configs/diffuser_test/ogbench", config_name="ogbench_segment_stitch", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    planner_name_suffix = "_subgoal_constrained" if _subgoal_filter_enabled(args) else ""
    if args.enable_wandb and args.mode in ["train", "inference"]:
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=f"{args.run_alias}{planner_name_suffix}_{args.mode}",
            config=OmegaConf.to_container(args, resolve=True),
        )

    set_seed(args.seed)

    save_path_suffix = "_SubgoalConstrained" if _subgoal_filter_enabled(args) else ""
    save_path = (
        f"results/{args.pipeline_name}/"
        f"{args.task.env_name}_LongH{args.task.planner_horizon}_Seg{args.stitch.segment_model_horizon}{save_path_suffix}/"
    )
    lowctrl_save_path = f"results/{args.pipeline_name}/{args.task.env_name}_LOW/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(lowctrl_save_path, exist_ok=True)
    planner_ckpt_stem = _planner_ckpt_stem(args)

    env, dataset, _ = ogbench.make_env_and_datasets(args.task.env_name, compact_dataset=True)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    goal_indices = resolve_goal_indices(args.task, obs_dim)
    goal_dim = int(goal_indices.size)
    goal_idx_tensor = torch.as_tensor(goal_indices, dtype=torch.long)

    jump_steps = int(args.task.get("jump_steps", 1))
    total_model_horizon = (int(args.task.planner_horizon) - 1) // jump_steps + 1
    segment_model_horizon = int(args.stitch.segment_model_horizon)
    segment_raw_horizon = 1 + (segment_model_horizon - 1) * jump_steps
    analysis_log_dir = None
    if _subgoal_filter_enabled(args):
        analysis_log_dir = _build_subgoal_analysis_log_dir(save_path)
        os.makedirs(analysis_log_dir, exist_ok=True)

    planner_dataset = OGBenchDataset(
        dataset,
        horizon=segment_raw_horizon,
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
    nn_condition_planner = SegmentBoundaryCondition(
        obs_dim=obs_dim,
        overlap_steps=args.stitch.overlap_steps,
        emb_dim=args.planner_emb_dim,
        hidden_dim=args.stitch.condition_hidden_dim,
        dropout=0.0,
    )
    print("=============== Parameter Report of Planner ==================================")
    report_parameters(nn_diffusion_planner)
    print("--------------- Condition Encoder --------------------------------------------")
    report_parameters(nn_condition_planner)
    print("==============================================================================")

    classifier = None
    if args.enable_distance_guidance:
        nn_classifier = MLPNNClassifier(1, 1, 1, [8, 8])
        distance_dims = []
        for idx in range(object_num.get(args.task.env_name, 1)):
            distance_dims.extend(list(range(-9 * idx - 9, -9 * idx - 6)))
        classifier = GCDistance(
            nn_classifier,
            device=args.device,
            distance_dims=torch.tensor(distance_dims, device=args.device),
        )

    fix_mask = torch.zeros((segment_model_horizon, obs_dim))
    loss_weight = torch.ones((segment_model_horizon, obs_dim))
    planner = SegmentStitchDiscreteDiffusionSDE(
        nn_diffusion_planner,
        nn_condition=nn_condition_planner,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=classifier,
        diffusion_steps=args.planner_diffusion_steps,
        ema_rate=args.planner_ema_rate,
        device=args.device,
        predict_noise=args.planner_predict_noise,
        noise_schedule="linear",
        overlap_steps=args.stitch.overlap_steps,
        condition_guidance_scale=args.stitch.condition_guidance_scale,
        train_overlap_prob=args.training_boundary.overlap_prob,
        train_side_drop_prob=args.training_boundary.side_drop_prob,
        gsc_inner_loops=args.stitch.gsc_inner_loops,
        gsc_keep_ratio=args.stitch.gsc_keep_ratio,
        gsc_filter_start=args.stitch.gsc_filter_start,
        gsc_inversion_ratio=args.stitch.gsc_inversion_ratio,
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

        if _subgoal_filter_enabled(args):
            if args.use_diffusion_invdyn:
                raise ValueError(
                    "subgoal_filter.enable=True currently requires use_diffusion_invdyn=False "
                    "because the filter relies on the GCIQL value function."
                )

            trajs = _get_trajectory_obs_list(planner_dataset)
            subgoal_results = detect_subgoals_for_all_trajectories(
                invdyn,
                trajectories=trajs,
                goal_indices=goal_indices,
                k_goals=args.subgoal_filter.k_goals,
                c_lookback=args.subgoal_filter.c_lookback,
                device=args.device,
                min_traj_length=args.subgoal_filter.min_traj_length,
                min_subgoal_dist=args.subgoal_filter.min_subgoal_dist,
                max_subgoals_per_traj=args.subgoal_filter.max_subgoals_per_traj,
            )

            kept_indices, removed, cross_ratios = build_subgoal_filtered_indices(
                planner_dataset,
                subgoal_results,
                horizon=segment_raw_horizon,
                max_cross_ratio=args.subgoal_filter.max_cross_ratio,
            )
            subgoal_counts = np.array([len(x["valid_subgoals"]) for x in subgoal_results], dtype=np.float32)
            keep_ratio = len(kept_indices) / max(1, len(planner_dataset))

            _save_subgoal_analysis_summary_json(
                subgoal_results=subgoal_results,
                cross_ratios=cross_ratios,
                cross_ratio_threshold=args.subgoal_filter.max_cross_ratio,
                output_path=os.path.join(analysis_log_dir, "subgoal_filter_summary.json"),
            )
            _save_histogram_png_and_json(
                cross_ratios,
                output_prefix=os.path.join(analysis_log_dir, "cross_ratio_hist"),
                bins=args.subgoal_filter.analysis.histogram_bins_cross_ratio,
                hist_range=(0.0, 1.0),
                title="Cross Ratio Histogram",
            )
            _save_histogram_png_and_json(
                subgoal_counts,
                output_prefix=os.path.join(analysis_log_dir, "subgoal_count_hist"),
                bins=args.subgoal_filter.analysis.histogram_bins_subgoal_count,
                title="Subgoal Count Histogram",
            )

            print(f"[SubgoalFilter] Saved local artifacts to: {analysis_log_dir}")
            print(
                f"[PlannerData][SubgoalFilter] keep={len(kept_indices)} / total={len(planner_dataset)} "
                f"(keep_ratio={keep_ratio:.4f}), removed={removed}, horizon={segment_raw_horizon}"
            )

            if args.enable_wandb:
                wandb.log(
                    {
                        "subgoal_filter/summary": build_subgoal_analysis_tables(
                            planner_dataset,
                            subgoal_results,
                            cross_ratios,
                            args.subgoal_filter.max_cross_ratio,
                        ),
                        "subgoal_filter/cross_ratio_hist": _to_wandb_histogram(
                            cross_ratios,
                            bins=args.subgoal_filter.analysis.histogram_bins_cross_ratio,
                            hist_range=(0.0, 1.0),
                        ),
                        "subgoal_filter/subgoal_count_hist": _to_wandb_histogram(
                            subgoal_counts,
                            bins=args.subgoal_filter.analysis.histogram_bins_subgoal_count,
                        ),
                        "subgoal_filter/keep_ratio": keep_ratio,
                        "subgoal_filter/removed": removed,
                    },
                    step=0,
                )

            sampling_strategy = str(args.subgoal_filter.sampling.strategy)
            print(f"[PlannerData][SubgoalFilter] sampling_strategy={sampling_strategy}")

            if sampling_strategy == "hard":
                if keep_ratio < args.subgoal_filter.min_keep_ratio:
                    raise ValueError(
                        f"Filtered samples too few: keep_ratio={keep_ratio:.4f} < "
                        f"min_keep_ratio={args.subgoal_filter.min_keep_ratio}. "
                        "Please relax max_cross_ratio or subgoal_filter params."
                    )

                filtered_planner_dataset = Subset(planner_dataset, kept_indices)
                planner_dataloader = DataLoader(
                    filtered_planner_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True,
                )
            elif sampling_strategy == "soft":
                soft_weights = build_soft_sampling_weights(
                    cross_ratios,
                    alpha=args.subgoal_filter.sampling.soft_alpha,
                    min_weight=args.subgoal_filter.sampling.soft_min_weight,
                )
                sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(soft_weights),
                    num_samples=len(soft_weights),
                    replacement=True,
                )
                planner_dataloader = DataLoader(
                    planner_dataset,
                    batch_size=args.batch_size,
                    sampler=sampler,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True,
                )
                _save_histogram_png_and_json(
                    soft_weights,
                    output_prefix=os.path.join(analysis_log_dir, "soft_weight_hist"),
                    bins=args.subgoal_filter.analysis.histogram_bins_soft_weight,
                    hist_range=(0.0, 1.0),
                    title="Soft Sampling Weight Histogram",
                )
                if args.enable_wandb:
                    wandb.log(
                        {
                            "subgoal_filter/soft_weight_hist": _to_wandb_histogram(
                                soft_weights,
                                bins=args.subgoal_filter.analysis.histogram_bins_soft_weight,
                                hist_range=(0.0, 1.0),
                            )
                        },
                        step=0,
                    )
            else:
                raise ValueError(f"Invalid subgoal_filter.sampling.strategy: {sampling_strategy}")

        n_gradient_step = 0
        log = {"gradient_steps": 0, "avg_loss_planner": 0.0}
        pbar = tqdm(total=max(1, args.planner_diffusion_gradient_steps // args.log_interval))

        for planner_batch in loop_dataloader(planner_dataloader):
            planner_horizon_data = planner_batch["obs"]["state"].to(args.device)
            log["avg_loss_planner"] += planner.update(planner_horizon_data)["loss"]
            planner_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                out = {"gradient_steps": n_gradient_step + 1}
                for key, value in log.items():
                    if key != "gradient_steps":
                        out[key] = value / args.log_interval
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
                    (
                        eval_info,
                        _,
                        cur_renders,
                        trajs_planned_3d,
                        trajs_segment_3d,
                        trajs_actual_3d,
                    ) = segment_stitch_evaluate(
                        diffusions_model=planner,
                        mode=args.low_controller_mode,
                        low_controller=policy if args.use_diffusion_invdyn else invdyn,
                        env=env,
                        normalizer=planner_dataset.get_normalizer(),
                        task_id=task_id,
                        horizon=total_model_horizon,
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        config=args,
                        num_eval_episodes=args.num_eval_episodes,
                        num_video_episodes=args.num_video_episodes,
                        video_frame_skip=args.video_frame_skip,
                    )
                    renders.extend(cur_renders)
                    if args.num_video_episodes > 0 and trajs_planned_3d and trajs_segment_3d and trajs_actual_3d:
                        overall_trajectories_3d.append(trajs_planned_3d[0])
                        overall_segment_trajectories_3d.append(trajs_segment_3d[0])
                        overall_actual_trajectories_3d.append(trajs_actual_3d[0])
                    metric_names = ["success", "selected_overlap_score", "selected_gsc_score"]
                    eval_metrics.update(
                        {f"evaluation/{task_name}_{k}": v for k, v in eval_info.items() if k in metric_names}
                    )
                    for key, value in eval_info.items():
                        if key in metric_names:
                            overall_metrics[key].append(value)

                for key, value in overall_metrics.items():
                    eval_metrics[f"evaluation/overall_{key}"] = float(np.mean(value))

                if (
                    args.num_video_episodes > 0
                    and overall_trajectories_3d
                    and overall_segment_trajectories_3d
                    and overall_actual_trajectories_3d
                ):
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics["video"] = video
                    trajs_planned_3d = _stack_3d_trajectories_with_padding(overall_trajectories_3d)
                    trajs_segment_3d = _stack_segment_3d_trajectories_with_padding(overall_segment_trajectories_3d)
                    trajs_actual_3d = _stack_3d_trajectories_with_padding(overall_actual_trajectories_3d)
                    eval_metrics["3d_trajectories"] = visualize_3d_trajectories_wandb(
                        planned_traj=trajs_planned_3d,
                        planned_segment_traj=trajs_segment_3d,
                        actual_traj=trajs_actual_3d,
                        n_cols=num_tasks,
                    )

                if args.enable_wandb:
                    wandb.log(eval_metrics, step=n_gradient_step + 1)
                planner.train()

            if (n_gradient_step + 1) % args.save_interval == 0:
                planner.save(os.path.join(save_path, f"{planner_ckpt_stem}_ckpt_{n_gradient_step + 1}.pt"))
                planner.save(os.path.join(save_path, f"{planner_ckpt_stem}_ckpt_latest.pt"))

            n_gradient_step += 1
            if n_gradient_step >= args.planner_diffusion_gradient_steps:
                print("===================== Training Finished =====================")
                break

        planner.save(os.path.join(save_path, f"{planner_ckpt_stem}_ckpt_latest.pt"))

    elif args.mode == "inference":
        planner.load(os.path.join(save_path, f"{planner_ckpt_stem}_ckpt_{args.planner_ckpt}.pt"))
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
            (
                eval_info,
                _,
                cur_renders,
                trajs_planned_3d,
                trajs_segment_3d,
                trajs_actual_3d,
            ) = segment_stitch_evaluate(
                diffusions_model=planner,
                mode=args.low_controller_mode,
                low_controller=policy if args.use_diffusion_invdyn else invdyn,
                env=env,
                normalizer=planner_dataset.get_normalizer(),
                task_id=task_id,
                horizon=total_model_horizon,
                obs_dim=obs_dim,
                act_dim=act_dim,
                config=args,
                num_eval_episodes=args.num_eval_episodes,
                num_video_episodes=args.num_video_episodes,
                video_frame_skip=args.video_frame_skip,
            )
            renders.extend(cur_renders)
            if args.num_video_episodes > 0 and trajs_planned_3d and trajs_segment_3d and trajs_actual_3d:
                overall_trajectories_3d.append(trajs_planned_3d[0])
                overall_segment_trajectories_3d.append(trajs_segment_3d[0])
                overall_actual_trajectories_3d.append(trajs_actual_3d[0])
            metric_names = ["success", "selected_overlap_score", "selected_gsc_score"]
            eval_metrics.update(
                {f"evaluation/{task_name}_{k}": v for k, v in eval_info.items() if k in metric_names}
            )
            for key, value in eval_info.items():
                if key in metric_names:
                    overall_metrics[key].append(value)

        for key, value in overall_metrics.items():
            eval_metrics[f"evaluation/overall_{key}"] = float(np.mean(value))

        if (
            args.num_video_episodes > 0
            and overall_trajectories_3d
            and overall_segment_trajectories_3d
            and overall_actual_trajectories_3d
        ):
            eval_metrics["video"] = get_wandb_video(renders=renders, n_cols=num_tasks)
            trajs_planned_3d = _stack_3d_trajectories_with_padding(overall_trajectories_3d)
            trajs_segment_3d = _stack_segment_3d_trajectories_with_padding(overall_segment_trajectories_3d)
            trajs_actual_3d = _stack_3d_trajectories_with_padding(overall_actual_trajectories_3d)
            eval_metrics["3d_trajectories"] = visualize_3d_trajectories_wandb(
                planned_traj=trajs_planned_3d,
                planned_segment_traj=trajs_segment_3d,
                actual_traj=trajs_actual_3d,
                n_cols=num_tasks,
            )

        if args.enable_wandb:
            wandb.log(eval_metrics, step=1)
        else:
            print(eval_metrics)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
