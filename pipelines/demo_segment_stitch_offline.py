import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ogbench
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import set_seed
from cleandiffuser_sup.diffusion import SegmentStitchDiscreteDiffusionSDE
from cleandiffuser_sup.nn_condition import SegmentBoundaryCondition
from cleandiffuser_sup.datasets.ogbench_dataset import OGBenchDataset

NUM_VISUAL_PAIRS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Offline segment-stitch demo on cube-single-play-v0.")
    parser.add_argument("--mode", type=str, choices=["train", "visualize", "train_and_visualize"], default="train_and_visualize")
    parser.add_argument("--env-name", type=str, default="cube-single-play-v0")
    parser.add_argument("--total-horizon", type=int, default=80)
    parser.add_argument("--num-segments", type=int, default=2)
    parser.add_argument("--overlap-steps", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-window-index", type=int, default=0)
    parser.add_argument("--planner-num-candidates", type=int, default=8)
    parser.add_argument("--planner-diffusion-steps", type=int, default=1000)
    parser.add_argument("--planner-sampling-steps", type=int, default=50)
    parser.add_argument("--planner-solver", type=str, default="ddim")
    parser.add_argument("--planner-use-ema", action="store_true", default=True)
    parser.add_argument("--planner-temperature", type=float, default=1.0)
    parser.add_argument("--planner-emb-dim", type=int, default=128)
    parser.add_argument("--planner-d-model", type=int, default=256)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--condition-hidden-dim", type=int, default=256)
    parser.add_argument("--segment-stitch-checkpoint", type=str, default="")
    parser.add_argument("--train-steps", type=int, default=100000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--train-learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-log-interval", type=int, default=1000)
    parser.add_argument("--train-save-interval", type=int, default=100000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--merge-mode", type=str, default="exp")
    parser.add_argument("--merge-beta", type=float, default=3.0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/zxh/Workspace/Diffusion-long-H/results/multiseg_diffuser",
    )
    parser.add_argument("--run-alias", type=str, default="demo_multiseg")
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/zxh/Workspace/Diffusion-long-H/results/multiseg_diffuser/cube_single_offline_demo_h80.png",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def compute_segment_horizon(total_horizon: int, num_segments: int, overlap_steps: int) -> int:
    numerator = total_horizon + (num_segments - 1) * overlap_steps
    if numerator % num_segments != 0:
        raise ValueError(
            "total_horizon + (num_segments - 1) * overlap_steps must be divisible by num_segments. "
            f"Got total_horizon={total_horizon}, num_segments={num_segments}, overlap_steps={overlap_steps}."
        )
    return numerator // num_segments


def get_checkpoint_dir(results_dir: str, env_name: str, segment_horizon: int) -> Path:
    return Path(results_dir) / f"{env_name}_Multi_SegH{segment_horizon}"


def get_valid_windows(terminals: np.ndarray, horizon: int) -> List[Tuple[int, int]]:
    valid = []
    path_start = 0
    for idx in range(terminals.shape[0]):
        if terminals[idx]:
            path_end = idx + 1
            path_length = path_end - path_start
            if path_length >= horizon:
                for start in range(path_start, path_end - horizon + 1):
                    valid.append((start, start + horizon))
            path_start = path_end
    return valid


def select_window(observations: np.ndarray, terminals: np.ndarray, horizon: int, window_index: int) -> Tuple[np.ndarray, int, int]:
    valid_windows = get_valid_windows(terminals, horizon)
    if len(valid_windows) == 0:
        raise RuntimeError(f"No valid trajectory window of length {horizon} found.")
    window_index = int(np.clip(window_index, 0, len(valid_windows) - 1))
    start, end = valid_windows[window_index]
    return observations[start:end], start, end


def select_window_batch(
    observations: np.ndarray,
    terminals: np.ndarray,
    horizon: int,
    window_index: int,
    num_windows: int,
):
    valid_windows = get_valid_windows(terminals, horizon)
    if len(valid_windows) == 0:
        raise RuntimeError(f"No valid trajectory window of length {horizon} found.")

    window_index = int(np.clip(window_index, 0, len(valid_windows) - 1))
    available_ids = np.arange(window_index, len(valid_windows), dtype=np.int32)
    if available_ids.shape[0] == 0:
        available_ids = np.arange(len(valid_windows), dtype=np.int32)

    if available_ids.shape[0] >= num_windows:
        chosen_pos = np.linspace(0, available_ids.shape[0] - 1, num=num_windows, dtype=int)
        chosen_ids = available_ids[chosen_pos]
    else:
        pad_value = int(available_ids[-1])
        chosen_ids = np.pad(
            available_ids,
            (0, num_windows - available_ids.shape[0]),
            mode="constant",
            constant_values=pad_value,
        )

    windows = []
    for selected_id in chosen_ids.tolist():
        start, end = valid_windows[selected_id]
        windows.append(
            {
                "window_id": int(selected_id),
                "gt_window": observations[start:end],
                "start_idx": int(start),
                "end_idx": int(end),
            }
        )
    return windows


def build_planner(args, obs_dim: int, device: str, segment_horizon: int):
    nn_diffusion = DiT1d(
        obs_dim,
        emb_dim=args.planner_emb_dim,
        d_model=args.planner_d_model,
        n_heads=args.planner_d_model // 64,
        depth=args.planner_depth,
        timestep_emb_type="fourier",
    )
    nn_condition = SegmentBoundaryCondition(
        obs_dim=obs_dim,
        overlap_steps=args.overlap_steps,
        emb_dim=args.planner_emb_dim,
        hidden_dim=args.condition_hidden_dim,
        dropout=0.0,
    )
    planner = SegmentStitchDiscreteDiffusionSDE(
        nn_diffusion,
        nn_condition=nn_condition,
        fix_mask=torch.zeros((segment_horizon, obs_dim)),
        loss_weight=torch.ones((segment_horizon, obs_dim)),
        optim_params={"lr": args.train_learning_rate, "weight_decay": 1e-5},
        diffusion_steps=args.planner_diffusion_steps,
        ema_rate=0.995,
        device=device,
        predict_noise=True,
        noise_schedule="linear",
        overlap_steps=args.overlap_steps,
    )
    return planner


def zero_module(module: torch.nn.Module):
    for param in module.parameters():
        torch.nn.init.zeros_(param)


def load_segment_stitch_checkpoint(planner, ckpt_path: str) -> bool:
    if not ckpt_path:
        return False
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        return False
    planner.load(ckpt_path)
    return True

def train_planner(
    args,
    planner: SegmentStitchDiscreteDiffusionSDE,
    dataset,
    segment_horizon: int,
    checkpoint_dir: Path,
):
    if args.train_steps <= 0:
        return None

    train_dataset = OGBenchDataset(
        dataset,
        horizon=segment_horizon,
        max_path_length=1001,
        jump_steps=1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(planner.device != "cpu"),
        drop_last=True,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    planner.train()
    scheduler = CosineAnnealingLR(planner.optimizer, args.train_steps)
    running_loss = 0.0
    progress = tqdm(total=args.train_steps, desc="segment-stitch-train")

    for step, batch in enumerate(loop_dataloader(train_loader), start=1):
        obs_batch = batch["obs"]["state"].to(planner.device)
        log = planner.update(obs_batch)
        scheduler.step()
        running_loss += float(log["loss"])
        progress.update(1)

        if step % args.train_log_interval == 0:
            avg_loss = running_loss / float(args.train_log_interval)
            print(f"[train] step={step} avg_loss={avg_loss:.6f}")
            running_loss = 0.0

        if step % args.train_save_interval == 0:
            planner.save(checkpoint_dir / f"{args.run_alias}_planner_ckpt_{step}.pt")
            planner.save(checkpoint_dir / f"{args.run_alias}_planner_ckpt_latest.pt")

        if step >= args.train_steps:
            break

    progress.close()
    final_ckpt = checkpoint_dir / f"{args.run_alias}_planner_ckpt_latest.pt"
    planner.save(final_ckpt)
    planner.eval()
    return final_ckpt


def sample_trajectory(
    planner: SegmentStitchDiscreteDiffusionSDE,
    normalizer,
    start_obs: np.ndarray,
    goal_obs: np.ndarray,
    total_horizon: int,
    strategy: str,
    num_segments: int,
    num_candidates: int,
    sample_steps: int,
    solver: str,
    temperature: float,
    top_n: int,
    merge_mode: str,
    merge_beta: float,
    use_ema: bool,
):
    start_state = torch.as_tensor(
        normalizer.normalize(start_obs[None]),
        device=planner.device,
        dtype=torch.float32,
    ).repeat(num_candidates, 1)
    goal_state = torch.as_tensor(
        normalizer.normalize(goal_obs[None]),
        device=planner.device,
        dtype=torch.float32,
    ).repeat(num_candidates, 1)

    segment_list, sample_log = planner.sample_segment_chain(
        start_state=start_state,
        goal_state=goal_state,
        num_segments=num_segments,
        strategy=strategy,
        solver=solver,
        sample_steps=sample_steps,
        sample_step_schedule="uniform",
        use_ema=use_ema,
        temperature=temperature,
        condition_cg=None,
        w_cg=0.0,
    )

    segment_list_np = [normalizer.unnormalize(segment.detach().cpu().numpy()) for segment in segment_list]
    order, scores = planner.rank_overlap_mismatch(segment_list_np, overlap_steps=planner.overlap_steps)
    top_n = max(1, min(top_n, order.shape[0]))
    top_idx = order[:top_n]
    stitched = planner.blend_segments(
        [segment[top_idx] for segment in segment_list_np],
        overlap_steps=planner.overlap_steps,
        blend_mode=merge_mode,
        blend_beta=merge_beta,
        target_horizon=total_horizon,
    )
    chosen = stitched[0]

    extra = {
        "candidate_scores": scores,
        "top_idx": top_idx,
        "selected_score": float(scores[top_idx[0]]),
        "segment_list": segment_list_np,
        "gsc_scores": sample_log.get("gsc_scores", None),
    }
    return chosen, extra


def plot_object_xyz(
    result_rows,
    meta_text: str,
    output_path: str,
):
    num_rows = len(result_rows)
    fig, axes = plt.subplots(
        num_rows,
        2,
        figsize=(14, 5 * num_rows),
        subplot_kw={"projection": "3d"},
    )
    if num_rows == 1:
        axes = np.asarray([axes])

    all_xyz = np.concatenate(
        [
            xyz
            for row in result_rows
            for xyz in (row["gt_xyz"], row["interleave_xyz"], row["gsc_xyz"])
        ],
        axis=0,
    )
    xyz_min = all_xyz.min(axis=0)
    xyz_max = all_xyz.max(axis=0)
    xyz_margin = np.maximum((xyz_max - xyz_min) * 0.08, 1e-3)

    items = [
        ("Interleave", "interleave_xyz", "interleave_score", "#1f77b4"),
        ("GSC", "gsc_xyz", "gsc_score", "#d62728"),
    ]

    for row_idx, row in enumerate(result_rows):
        for col_idx, (title, xyz_key, score_key, color) in enumerate(items):
            ax = axes[row_idx, col_idx]
            gt_xyz = row["gt_xyz"]
            pred_xyz = row[xyz_key]

            ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], "--", color="black", linewidth=1.4, label="Ground Truth")
            ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], color=color, linewidth=2.0, label=title)
            ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], color="green", s=36, label="GT Start")
            ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], color="red", s=36, label="GT Goal")
            ax.scatter(pred_xyz[0, 0], pred_xyz[0, 1], pred_xyz[0, 2], facecolors="none", edgecolors=color, s=42, linewidths=1.2)
            ax.scatter(pred_xyz[-1, 0], pred_xyz[-1, 1], pred_xyz[-1, 2], facecolors="none", edgecolors=color, s=42, linewidths=1.2)
            ax.set_title(
                f"Pair {row_idx + 1} | {title} | window=[{row['start_idx']}, {row['end_idx']})\n"
                f"score={row[score_key]:.6f}"
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(xyz_min[0] - xyz_margin[0], xyz_max[0] + xyz_margin[0])
            ax.set_ylim(xyz_min[1] - xyz_margin[1], xyz_max[1] + xyz_margin[1])
            ax.set_zlim(xyz_min[2] - xyz_margin[2], xyz_max[2] + xyz_margin[2])
            ax.view_init(elev=25, azim=-55)
            if row_idx == 0:
                ax.legend(loc="best")

    fig.suptitle("cube-single-play-v0 offline segment stitch demo\n" + meta_text, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.sample_seed)
    device = resolve_device(args.device)

    segment_horizon = compute_segment_horizon(args.total_horizon, args.num_segments, args.overlap_steps)
    checkpoint_dir = get_checkpoint_dir(args.results_dir, args.env_name, segment_horizon)
    env, dataset, _ = ogbench.make_env_and_datasets(args.env_name, compact_dataset=True)
    del env

    observations = dataset["observations"].astype(np.float32)
    terminals = dataset["terminals"].astype(bool)
    window_batch = select_window_batch(
        observations,
        terminals,
        horizon=args.total_horizon,
        window_index=args.sample_window_index,
        num_windows=NUM_VISUAL_PAIRS,
    )

    normalizer = OGBenchDataset(
        dataset,
        horizon=segment_horizon,
        max_path_length=1001,
        jump_steps=1,
    ).get_normalizer()
    planner = build_planner(args, obs_dim=observations.shape[-1], device=device, segment_horizon=segment_horizon)

    load_mode = "random_init"
    checkpoint_path = args.segment_stitch_checkpoint
    if not checkpoint_path:
        latest_ckpt = checkpoint_dir / f"{args.run_alias}_planner_ckpt_latest.pt"
        if latest_ckpt.exists():
            checkpoint_path = str(latest_ckpt)

    if checkpoint_path and load_segment_stitch_checkpoint(planner, checkpoint_path):
        load_mode = f"segment_stitch:{checkpoint_path}"
    elif args.mode == "visualize":
        raise FileNotFoundError(
            "No segment-stitch checkpoint found for visualize mode. "
            "Run with --mode train_and_visualize first, or pass --segment-stitch-checkpoint."
        )

    trained_ckpt = None
    if args.mode in ["train", "train_and_visualize"]:
        trained_ckpt = train_planner(
            args=args,
            planner=planner,
            dataset=dataset,
            segment_horizon=segment_horizon,
            checkpoint_dir=checkpoint_dir,
        )
        if trained_ckpt is not None:
            load_mode = f"trained:{trained_ckpt}"
            planner.load(str(trained_ckpt))

    if args.mode == "train":
        print(f"Training finished. Latest checkpoint: {trained_ckpt}")
        return

    planner.eval()

    if args.mode == "visualize" and checkpoint_path:
        load_mode = f"segment_stitch:{checkpoint_path}"

    result_rows = []
    for row_idx, window_info in enumerate(tqdm(window_batch, desc="segment-stitch-visualize"), start=1):
        gt_window = window_info["gt_window"]

        interleave_traj, interleave_info = sample_trajectory(
            planner=planner,
            normalizer=normalizer,
            start_obs=gt_window[0],
            goal_obs=gt_window[-1],
            total_horizon=args.total_horizon,
            strategy="interleave",
            num_segments=args.num_segments,
            num_candidates=args.planner_num_candidates,
            sample_steps=args.planner_sampling_steps,
            solver=args.planner_solver,
            temperature=args.planner_temperature,
            top_n=args.top_n,
            merge_mode=args.merge_mode,
            merge_beta=args.merge_beta,
            use_ema=args.planner_use_ema,
        )
        gsc_traj, gsc_info = sample_trajectory(
            planner=planner,
            normalizer=normalizer,
            start_obs=gt_window[0],
            goal_obs=gt_window[-1],
            total_horizon=args.total_horizon,
            strategy="gsc",
            num_segments=args.num_segments,
            num_candidates=args.planner_num_candidates,
            sample_steps=args.planner_sampling_steps,
            solver=args.planner_solver,
            temperature=args.planner_temperature,
            top_n=args.top_n,
            merge_mode=args.merge_mode,
            merge_beta=args.merge_beta,
            use_ema=args.planner_use_ema,
        )

        result_rows.append(
            {
                "pair_id": row_idx,
                "window_id": window_info["window_id"],
                "start_idx": window_info["start_idx"],
                "end_idx": window_info["end_idx"],
                "gt_traj": gt_window,
                "interleave_traj": interleave_traj,
                "gsc_traj": gsc_traj,
                "gt_xyz": gt_window[:, -9:-6],
                "interleave_xyz": interleave_traj[:, -9:-6],
                "gsc_xyz": gsc_traj[:, -9:-6],
                "interleave_score": interleave_info["selected_score"],
                "gsc_score": gsc_info["selected_score"],
            }
        )

    meta_text = (
        f"pairs={len(result_rows)}, total_horizon={args.total_horizon}, "
        f"segments={args.num_segments}, overlap={args.overlap_steps}\n"
        f"load={load_mode}\n"
        f"same start/goal pair is used within each row for interleave and gsc"
    )
    plot_object_xyz(
        result_rows=result_rows,
        meta_text=meta_text,
        output_path=args.output_path,
    )

    np.savez(
        Path(args.output_path).with_suffix(".npz"),
        gt_traj=np.stack([row["gt_traj"] for row in result_rows], axis=0),
        interleave_traj=np.stack([row["interleave_traj"] for row in result_rows], axis=0),
        gsc_traj=np.stack([row["gsc_traj"] for row in result_rows], axis=0),
        gt_xyz=np.stack([row["gt_xyz"] for row in result_rows], axis=0),
        interleave_xyz=np.stack([row["interleave_xyz"] for row in result_rows], axis=0),
        gsc_xyz=np.stack([row["gsc_xyz"] for row in result_rows], axis=0),
        window_ids=np.asarray([row["window_id"] for row in result_rows], dtype=np.int32),
        start_indices=np.asarray([row["start_idx"] for row in result_rows], dtype=np.int32),
        end_indices=np.asarray([row["end_idx"] for row in result_rows], dtype=np.int32),
        interleave_scores=np.asarray([row["interleave_score"] for row in result_rows], dtype=np.float32),
        gsc_scores=np.asarray([row["gsc_score"] for row in result_rows], dtype=np.float32),
    )

    print(f"Saved figure to: {args.output_path}")
    print(f"Saved trajectory arrays to: {Path(args.output_path).with_suffix('.npz')}")
    print(f"Selected {len(result_rows)} dataset windows for joint comparison, total_horizon={args.total_horizon}")
    print(f"Load mode: {load_mode}")
    if trained_ckpt is not None:
        print(f"Checkpoint saved to: {trained_ckpt}")
    for row in result_rows:
        print(
            f"Pair {row['pair_id']}: window_id={row['window_id']} "
            f"start={row['start_idx']} end={row['end_idx']} "
            f"interleave={row['interleave_score']:.6f} gsc={row['gsc_score']:.6f}"
        )


if __name__ == "__main__":
    main()
