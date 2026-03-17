import os
import math
import uuid
import json
import hydra
import wandb
import ogbench
import numpy as np
from datetime import datetime
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters, set_seed
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser_sup.lowcontrol.gciql_inv import GCIQLAgent
from cleandiffuser_sup.datasets.ogbench_dataset import OGBenchDataset, GCDataset
from evaluate import single_layer_evaluate
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


def _get_trajectory_obs_list(planner_dataset):
    trajs = []
    for path_idx, length in enumerate(planner_dataset.path_lengths):
        if length <= 0:
            continue
        trajs.append(planner_dataset.seq_obs[path_idx, :length].copy())
    return trajs


def detect_subgoals_for_all_trajectories(
    agent,
    trajectories,
    goal_indices,
    k_goals,
    c_lookback,
    device,
    min_traj_length=20,
    min_subgoal_dist=30,
    max_subgoals_per_traj=6,
):
    results = []
    obs_goal_indices = np.asarray(goal_indices, dtype=np.int64)

    for traj in trajectories:
        T = len(traj)
        if T < min_traj_length:
            results.append({"T": T, "valid_subgoals": np.array([], dtype=np.int64)})
            continue

        if obs_goal_indices.size > 0:
            obs_dim = int(traj.shape[-1])
            if np.min(obs_goal_indices) < 0 or np.max(obs_goal_indices) >= obs_dim:
                raise ValueError(
                    f"goal_indices out of bounds for observation dim: obs_dim={obs_dim}, "
                    f"goal_indices range=[{int(np.min(obs_goal_indices))}, {int(np.max(obs_goal_indices))}]"
                )

        L = T / max(k_goals, 1)
        subgoal_votes = np.zeros(T)
        opportunity_counts = np.zeros(T, dtype=np.float32)

        temporal_goal_indices = np.linspace(0, T - 1, k_goals + 1, dtype=int)[1:]
        lookback_steps = int(c_lookback * L)

        for g_idx in temporal_goal_indices:
            start_idx = max(0, g_idx - lookback_steps)
            opportunity_counts[start_idx:g_idx + 1] += 1.0
            indices = np.arange(start_idx, g_idx + 1)
            if len(indices) < 5:
                continue

            segment_obs = traj[indices]
            goal_obs = traj[g_idx]

            obs_tensor = torch.from_numpy(segment_obs).to(device).float()
            goal_tensor = (
                torch.from_numpy(goal_obs[obs_goal_indices])
                .to(device)
                .float()
                .unsqueeze(0)
                .expand(len(segment_obs), -1)
            )

            with torch.no_grad():
                values = agent.value_net(obs_tensor, goal_tensor).cpu().numpy().squeeze()

            smooth_win = min(5, len(values) // 2)
            if smooth_win > 1:
                values_smooth = uniform_filter1d(values, size=smooth_win)
            else:
                values_smooth = values

            val_range = np.max(values_smooth) - np.min(values_smooth) + 1e-6
            prominence = 0.05 * val_range
            peaks, _ = find_peaks(-values_smooth, prominence=prominence, distance=3)

            for p in peaks:
                subgoal_votes[indices[p]] += 1

        vote_density = np.convolve(subgoal_votes, np.ones(10), mode="same")
        final_peaks, _ = find_peaks(vote_density, height=0.5, distance=min_subgoal_dist)

        if len(final_peaks) > max_subgoals_per_traj:
            # Normalize by local "opportunity" counts to avoid unfairly penalizing boundary timesteps.
            norm_scores = vote_density[final_peaks] / (opportunity_counts[final_peaks] + 1e-6)
            top_idx = np.argsort(norm_scores)[-max_subgoals_per_traj:]
            final_peaks = np.sort(final_peaks[top_idx])

        results.append({"T": T, "valid_subgoals": final_peaks.astype(np.int64)})

    return results


def _to_wandb_histogram(values, bins=50, hist_range=None):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        arr = np.array([0.0], dtype=np.float32)
    if hist_range is not None:
        hist = np.histogram(arr, bins=bins, range=hist_range)
    else:
        hist = np.histogram(arr, bins=bins)
    return wandb.Histogram(np_histogram=hist)


def _save_histogram_png_and_json(values, output_prefix, bins=50, hist_range=None, title="histogram"):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        arr = np.array([0.0], dtype=np.float32)

    if hist_range is not None:
        counts, edges = np.histogram(arr, bins=bins, range=hist_range)
    else:
        counts, edges = np.histogram(arr, bins=bins)

    with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "title": title,
                "bins": int(bins),
                "hist_range": list(hist_range) if hist_range is not None else None,
                "counts": counts.astype(int).tolist(),
                "bin_edges": edges.astype(float).tolist(),
                "num_values": int(arr.size),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 4.5), dpi=120)
        plt.hist(arr, bins=bins, range=hist_range, color="#2b6cb0", alpha=0.85, edgecolor="white", linewidth=0.5)
        plt.title(title)
        plt.xlabel("value")
        plt.ylabel("count")
        plt.tight_layout()
        fig.savefig(f"{output_prefix}.png")
        plt.close(fig)
    except ImportError:
        print("[SubgoalAnalysis] matplotlib not found, skipped png export.")


def _save_subgoal_analysis_summary_json(subgoal_results, cross_ratios, cross_ratio_threshold, output_path):
    subgoal_counts = np.array([len(x["valid_subgoals"]) for x in subgoal_results], dtype=np.float32)
    summary = {
        "num_samples": int(len(cross_ratios)),
        "num_trajectories": int(len(subgoal_results)),
        "cross_ratio_threshold": float(cross_ratio_threshold),
        "cross_ratio_mean": float(np.mean(cross_ratios)) if len(cross_ratios) > 0 else 0.0,
        "cross_ratio_std": float(np.std(cross_ratios)) if len(cross_ratios) > 0 else 0.0,
        "cross_ratio_p50": float(np.percentile(cross_ratios, 50)) if len(cross_ratios) > 0 else 0.0,
        "cross_ratio_p90": float(np.percentile(cross_ratios, 90)) if len(cross_ratios) > 0 else 0.0,
        "cross_ratio_p99": float(np.percentile(cross_ratios, 99)) if len(cross_ratios) > 0 else 0.0,
        "cross_ratio_keep_ratio": float(np.mean(cross_ratios <= cross_ratio_threshold)) if len(cross_ratios) > 0 else 0.0,
        "subgoal_count_mean": float(np.mean(subgoal_counts)) if len(subgoal_counts) > 0 else 0.0,
        "subgoal_count_std": float(np.std(subgoal_counts)) if len(subgoal_counts) > 0 else 0.0,
        "subgoal_count_p50": float(np.percentile(subgoal_counts, 50)) if len(subgoal_counts) > 0 else 0.0,
        "subgoal_count_p90": float(np.percentile(subgoal_counts, 90)) if len(subgoal_counts) > 0 else 0.0,
        "subgoal_count_max": float(np.max(subgoal_counts)) if len(subgoal_counts) > 0 else 0.0,
        "subgoal_count_zero_ratio": float(np.mean(subgoal_counts == 0)) if len(subgoal_counts) > 0 else 0.0,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _build_analysis_log_dir(args):
    base_dir = f"logs/{args.pipeline_name}/{args.task.env_name}_SUBGOAL_CONSTRAINED_H{args.task.planner_horizon}"
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, run_time)


def build_subgoal_analysis_tables(
    planner_dataset,
    subgoal_results,
    cross_ratios,
    cross_ratio_threshold,
):
    _ = planner_dataset
    subgoal_counts = np.array([len(x["valid_subgoals"]) for x in subgoal_results], dtype=np.float32)
    summary_table = wandb.Table(columns=["metric", "value"])
    summary_table.add_data("num_samples", int(len(cross_ratios)))
    summary_table.add_data("num_trajectories", int(len(subgoal_results)))
    summary_table.add_data("cross_ratio_threshold", float(cross_ratio_threshold))
    summary_table.add_data("cross_ratio_mean", float(np.mean(cross_ratios)) if len(cross_ratios) > 0 else 0.0)
    summary_table.add_data("cross_ratio_std", float(np.std(cross_ratios)) if len(cross_ratios) > 0 else 0.0)
    summary_table.add_data("cross_ratio_p50", float(np.percentile(cross_ratios, 50)) if len(cross_ratios) > 0 else 0.0)
    summary_table.add_data("cross_ratio_p90", float(np.percentile(cross_ratios, 90)) if len(cross_ratios) > 0 else 0.0)
    summary_table.add_data("cross_ratio_p99", float(np.percentile(cross_ratios, 99)) if len(cross_ratios) > 0 else 0.0)
    summary_table.add_data("cross_ratio_keep_ratio", float(np.mean(cross_ratios <= cross_ratio_threshold)) if len(cross_ratios) > 0 else 0.0)
    summary_table.add_data("subgoal_count_mean", float(np.mean(subgoal_counts)) if len(subgoal_counts) > 0 else 0.0)
    summary_table.add_data("subgoal_count_std", float(np.std(subgoal_counts)) if len(subgoal_counts) > 0 else 0.0)
    summary_table.add_data("subgoal_count_p50", float(np.percentile(subgoal_counts, 50)) if len(subgoal_counts) > 0 else 0.0)
    summary_table.add_data("subgoal_count_p90", float(np.percentile(subgoal_counts, 90)) if len(subgoal_counts) > 0 else 0.0)
    summary_table.add_data("subgoal_count_max", float(np.max(subgoal_counts)) if len(subgoal_counts) > 0 else 0.0)
    summary_table.add_data("subgoal_count_zero_ratio", float(np.mean(subgoal_counts == 0)) if len(subgoal_counts) > 0 else 0.0)

    return summary_table


def _compute_cross_ratio(start, end_inclusive, subgoals):
    length = end_inclusive - start + 1
    if length <= 1 or len(subgoals) == 0:
        return 0.0

    boundaries = np.concatenate(([0], subgoals, [10 ** 9]))
    counts = defaultdict(int)
    for t in range(start, end_inclusive + 1):
        seg_id = np.searchsorted(boundaries, t, side="right") - 1
        counts[int(seg_id)] += 1

    dominant = max(counts.values()) if counts else length
    return 1.0 - dominant / float(length)


def build_subgoal_filtered_indices(planner_dataset, subgoal_results, horizon, max_cross_ratio):
    kept = []
    removed = 0
    cross_ratios = []

    for sample_idx, (path_idx, start, end) in enumerate(planner_dataset.indices):
        _ = end
        path_length = planner_dataset.path_lengths[path_idx]
        end_inclusive = min(start + horizon - 1, path_length - 1)
        subgoals = subgoal_results[path_idx]["valid_subgoals"]

        cross_ratio = _compute_cross_ratio(start, end_inclusive, subgoals)
        cross_ratios.append(cross_ratio)
        if cross_ratio <= max_cross_ratio:
            kept.append(sample_idx)
        else:
            removed += 1

    cross_ratios = np.array(cross_ratios, dtype=np.float32)
    return kept, removed, cross_ratios


def build_soft_sampling_weights(cross_ratios, alpha=6.0, min_weight=0.05):
    # Larger cross_ratio -> smaller weight, but keep a minimum probability floor.
    weights = np.exp(-alpha * cross_ratios)
    weights = np.clip(weights, min_weight, 1.0)
    return weights.astype(np.float64)


def train_low_controller_if_needed(args, invdyn, policy_dataloader, lowctrl_save_path, goal_idx_tensor):
    ckpt_name = f"{args.lowctrl_alias}_lowctrl_ckpt_latest.pt"
    default_ckpt_path = os.path.join(lowctrl_save_path, ckpt_name)

    if args.lowctrl_load_existing:
        load_path = args.lowctrl_load_path if args.lowctrl_load_path else default_ckpt_path
        if os.path.exists(load_path):
            print(f"[LowCtrl] Load existing checkpoint: {load_path}")
            invdyn.load(load_path)
            return
        print(f"[LowCtrl] Requested load path not found: {load_path}")
        if not args.lowctrl_train_if_missing:
            raise FileNotFoundError(load_path)

    print("[LowCtrl] Start training")
    invdyn.train()
    log = {
        "gradient_steps": 0,
        "policy_loss_value": 0.0,
        "policy_loss_critic": 0.0,
        "policy_loss_actor": 0.0,
    }

    pbar = tqdm(total=max(1, args.invdyn_gradient_steps // args.log_interval), desc="LowCtrl")

    for n_gradient_step, batch in enumerate(loop_dataloader(policy_dataloader), start=1):
        idx = goal_idx_tensor.to(batch["value_goals"].device)
        batch["value_goals"] = batch["value_goals"].index_select(-1, idx)
        batch["actor_goals"] = batch["actor_goals"].index_select(-1, idx)

        info = invdyn.update(batch)
        log["policy_loss_value"] += info["value/value_loss"]
        log["policy_loss_critic"] += info["critic/critic_loss"]
        log["policy_loss_actor"] += info["actor/actor_loss"]

        if n_gradient_step % args.log_interval == 0:
            out = {"gradient_steps": n_gradient_step}
            out.update({k: v / args.log_interval for k, v in log.items() if k != "gradient_steps"})
            print(out)
            if args.enable_wandb:
                wandb.log({f"lowctrl/{k}": v for k, v in out.items()}, step=n_gradient_step)
            pbar.update(1)
            log = {
                "gradient_steps": 0,
                "policy_loss_value": 0.0,
                "policy_loss_critic": 0.0,
                "policy_loss_actor": 0.0,
            }

        if n_gradient_step % args.save_interval == 0:
            invdyn.save(os.path.join(lowctrl_save_path, f"{args.lowctrl_alias}_lowctrl_ckpt_{n_gradient_step}.pt"))
            invdyn.save(default_ckpt_path)

        if n_gradient_step >= args.invdyn_gradient_steps:
            break

    invdyn.save(default_ckpt_path)
    print(f"[LowCtrl] Training done. Saved: {default_ckpt_path}")


@hydra.main(config_path="../configs/diffuser_test/ogbench", config_name="ogbench_subgoal_constrained", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"

    if args.enable_wandb and args.mode in ["train", "inference"]:
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.run_alias) + "_subgoal_constrained_" + str(args.mode),
            config=OmegaConf.to_container(args, resolve=True),
        )

    set_seed(args.seed)

    save_path = f"results/{args.pipeline_name}/{args.task.env_name}_SUBGOAL_CONSTRAINED_H{args.task.planner_horizon}/"
    lowctrl_save_path = f"results/{args.pipeline_name}/{args.task.env_name}_LOW/"
    analysis_log_dir = _build_analysis_log_dir(args)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(lowctrl_save_path, exist_ok=True)
    os.makedirs(analysis_log_dir, exist_ok=True)


    env, dataset, _ = ogbench.make_env_and_datasets(args.task.env_name, compact_dataset=True)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    jump_steps = args.task.get("jump_steps", 1)

    planner_dataset = OGBenchDataset(
        dataset,
        horizon=args.task.planner_horizon,
        max_path_length=args.task.max_path_length,
        jump_steps=jump_steps,
    )

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
    print(f"[LowCtrl] goal_dim={goal_dim}, goal_indices={goal_indices.tolist()} (from task.goal_dim)")

    if args.mode == "train":
        # Stage 1: train/load low-level value function.
        train_low_controller_if_needed(args, invdyn, policy_dataloader, lowctrl_save_path, goal_idx_tensor)
        invdyn.eval()

        # Stage 2: detect subgoals and filter planner dataset.
        trajs = _get_trajectory_obs_list(planner_dataset)
        subgoal_results = detect_subgoals_for_all_trajectories(
            invdyn,
            trajectories=trajs,
            goal_indices=goal_indices,
            k_goals=args.subgoal.k_goals,
            c_lookback=args.subgoal.c_lookback,
            device=args.device,
            min_traj_length=args.subgoal.min_traj_length,
            min_subgoal_dist=args.subgoal.min_subgoal_dist,
            max_subgoals_per_traj=args.subgoal.max_subgoals_per_traj,
        )

        kept_indices, removed, cross_ratios = build_subgoal_filtered_indices(
            planner_dataset,
            subgoal_results,
            horizon=args.task.planner_horizon,
            max_cross_ratio=args.subgoal.max_cross_ratio,
        )

        subgoal_counts = np.array([len(x["valid_subgoals"]) for x in subgoal_results], dtype=np.float32)
        _save_subgoal_analysis_summary_json(
            subgoal_results=subgoal_results,
            cross_ratios=cross_ratios,
            cross_ratio_threshold=args.subgoal.max_cross_ratio,
            output_path=os.path.join(analysis_log_dir, "subgoal_filter_summary.json"),
        )
        _save_histogram_png_and_json(
            cross_ratios,
            output_prefix=os.path.join(analysis_log_dir, "cross_ratio_hist"),
            bins=args.subgoal.analysis.histogram_bins_cross_ratio,
            hist_range=(0.0, 1.0),
            title="Cross Ratio Histogram",
        )
        _save_histogram_png_and_json(
            subgoal_counts,
            output_prefix=os.path.join(analysis_log_dir, "subgoal_count_hist"),
            bins=args.subgoal.analysis.histogram_bins_subgoal_count,
            title="Subgoal Count Histogram",
        )
        print(f"[SubgoalAnalysis] Saved local artifacts to: {analysis_log_dir}")

        keep_ratio = len(kept_indices) / max(1, len(planner_dataset))
        print(
            f"[PlannerData] keep={len(kept_indices)} / total={len(planner_dataset)} "
            f"(keep_ratio={keep_ratio:.4f}), removed={removed}"
        )

        sampling_strategy = args.subgoal.sampling.strategy
        print(f"[PlannerData] sampling_strategy={sampling_strategy}")

        if sampling_strategy == "hard":
            if keep_ratio < args.subgoal.min_keep_ratio:
                raise ValueError(
                    f"Filtered samples too few: keep_ratio={keep_ratio:.4f} < min_keep_ratio={args.subgoal.min_keep_ratio}. "
                    "Please relax max_cross_ratio or subgoal params."
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
                alpha=args.subgoal.sampling.soft_alpha,
                min_weight=args.subgoal.sampling.soft_min_weight,
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
                bins=args.subgoal.analysis.histogram_bins_soft_weight,
                hist_range=(0.0, 1.0),
                title="Soft Sampling Weight Histogram",
            )
        else:
            raise ValueError(f"Invalid subgoal.sampling.strategy: {sampling_strategy}")

        # Stage 3: train diffusion planner on filtered samples.
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

        model_horizon = (args.task.planner_horizon - 1) // jump_steps + 1
        fix_mask = torch.zeros((model_horizon, obs_dim))
        fix_mask[0, :obs_dim] = 1.0
        fix_mask[-1, :obs_dim] = 1.0

        loss_weight = torch.ones((model_horizon, obs_dim))

        planner = ContinuousDiffusionSDE(
            nn_diffusion_planner,
            nn_condition=None,
            fix_mask=fix_mask,
            loss_weight=loss_weight,
            classifier=None,
            ema_rate=args.planner_ema_rate,
            device=args.device,
            predict_noise=args.planner_predict_noise,
            noise_schedule="linear",
        )

        planner_scheduler = CosineAnnealingLR(planner.optimizer, args.planner_diffusion_gradient_steps)
        planner.train()

        pbar = tqdm(total=max(1, args.planner_diffusion_gradient_steps // args.log_interval), desc="Planner")
        log = {"gradient_steps": 0, "avg_loss_planner": 0.0}

        for n_gradient_step, planner_batch in enumerate(loop_dataloader(planner_dataloader), start=1):
            planner_horizon_obs = planner_batch["obs"]["state"].to(args.device)
            loss_info = planner.update(planner_horizon_obs)
            log["avg_loss_planner"] += loss_info["loss"]
            planner_scheduler.step()

            if n_gradient_step % args.log_interval == 0:
                out = {
                    "gradient_steps": n_gradient_step,
                    "avg_loss_planner": log["avg_loss_planner"] / args.log_interval,
                }
                print(out)
                if args.enable_wandb:
                    wandb.log(out, step=n_gradient_step)
                pbar.update(1)
                log = {"gradient_steps": 0, "avg_loss_planner": 0.0}

            if n_gradient_step % args.eval_interval == 0:
                planner.eval()
                invdyn.eval()

                renders = []
                eval_metrics = {}
                overall_metrics = defaultdict(list)
                overall_trajectories_3d = []
                overall_actual_trajectories_3d = []
                task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
                num_tasks = len(task_infos)

                for task_id in trange(1, num_tasks + 1):
                    task_name = task_infos[task_id - 1]["task_name"]
                    eval_info, _, cur_renders, trajs_planned_3d, trajs_actual_3d = single_layer_evaluate(
                        diffusions_model=planner,
                        mode=args.low_controller_mode,
                        low_controller=invdyn,
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
                        overall_actual_trajectories_3d.append(trajs_actual_3d[0])

                    metric_names = ["success"]
                    eval_metrics.update({f"evaluation/{task_name}_{k}": v for k, v in eval_info.items() if k in metric_names})
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)

                for k, v in overall_metrics.items():
                    eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

                if args.num_video_episodes > 0 and len(overall_trajectories_3d) > 0 and len(overall_actual_trajectories_3d) > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics["video"] = video

                    trajs_planned_3d = _stack_3d_trajectories_with_padding(overall_trajectories_3d)
                    trajs_actual_3d = _stack_3d_trajectories_with_padding(overall_actual_trajectories_3d)
                    trajs_image = visualize_3d_trajectories_wandb(
                        planned_traj=trajs_planned_3d,
                        actual_traj=trajs_actual_3d,
                        n_cols=num_tasks,
                    )
                    eval_metrics["3d_trajectories"] = trajs_image

                if args.enable_wandb:
                    wandb.log(eval_metrics, step=n_gradient_step)

                planner.train()

            if n_gradient_step % args.save_interval == 0:
                planner.save(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_{n_gradient_step}.pt"))
                planner.save(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_latest.pt"))

            if n_gradient_step >= args.planner_diffusion_gradient_steps:
                break

        planner.save(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_latest.pt"))
        print("===================== Training Finished =====================")

    elif args.mode == "inference":
        nn_diffusion_planner = DiT1d(
            obs_dim,
            emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model,
            n_heads=args.planner_d_model // 64,
            depth=args.planner_depth,
            timestep_emb_type="fourier",
        )

        model_horizon = (args.task.planner_horizon - 1) // jump_steps + 1
        fix_mask = torch.zeros((model_horizon, obs_dim))
        fix_mask[0, :obs_dim] = 1.0
        fix_mask[-1, :obs_dim] = 1.0
        loss_weight = torch.ones((model_horizon, obs_dim))

        planner = ContinuousDiffusionSDE(
            nn_diffusion_planner,
            nn_condition=None,
            fix_mask=fix_mask,
            loss_weight=loss_weight,
            classifier=None,
            ema_rate=args.planner_ema_rate,
            device=args.device,
            predict_noise=args.planner_predict_noise,
            noise_schedule="linear",
        )

        planner.load(os.path.join(save_path, f"{args.run_alias}_planner_ckpt_{args.planner_ckpt}.pt"))
        planner.eval()

        invdyn.load(os.path.join(save_path, f"{args.run_alias}_lowctrl_ckpt_{args.invdyn_ckpt}.pt"))
        invdyn.eval()

        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        overall_trajectories_3d = []
        overall_actual_trajectories_3d = []
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
        num_tasks = len(task_infos)

        for task_id in trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]["task_name"]
            eval_info, _, cur_renders, trajs_planned_3d, trajs_actual_3d = single_layer_evaluate(
                diffusions_model=planner,
                mode=args.low_controller_mode,
                low_controller=invdyn,
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
                overall_actual_trajectories_3d.append(trajs_actual_3d[0])

            metric_names = ["success"]
            eval_metrics.update({f"evaluation/{task_name}_{k}": v for k, v in eval_info.items() if k in metric_names})
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

        for k, v in overall_metrics.items():
            eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

        if args.num_video_episodes > 0 and len(overall_trajectories_3d) > 0 and len(overall_actual_trajectories_3d) > 0:
            video = get_wandb_video(renders=renders, n_cols=num_tasks)
            eval_metrics["video"] = video

            trajs_planned_3d = _stack_3d_trajectories_with_padding(overall_trajectories_3d)
            trajs_actual_3d = _stack_3d_trajectories_with_padding(overall_actual_trajectories_3d)
            trajs_image = visualize_3d_trajectories_wandb(
                planned_traj=trajs_planned_3d,
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
