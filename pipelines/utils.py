import random
import time
import uuid
import os
import json
import wandb
import numpy as np
import torch

import math
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from PIL import Image, ImageEnhance

# from cleandiffuser.env.wrapper import VideoRecordingWrapper


def parse_cfg(cfg_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    for k,v in cli.items():
        if v == None:
            cli[k] = True
    base.merge_with(cli)
    return base


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_goal_indices(task_cfg, obs_dim: int) -> np.ndarray:
    """Resolve task.goal_dim into explicit observation indices.

    Supported formats for task.goal_dim:
    - int N: use trailing N dims, i.e. [obs_dim-N, ..., obs_dim-1]
    - list/tuple of indices: explicit dims (supports negative indexing)
    - missing/None: use all dims [0, ..., obs_dim-1]
    """
    goal_spec = task_cfg.get("goal_dim", None)
    if goal_spec is None:
        return np.arange(obs_dim, dtype=np.int64)

    if isinstance(goal_spec, (int, np.integer)):
        if goal_spec <= 0 or goal_spec > obs_dim:
            raise ValueError(f"Invalid task.goal_dim={goal_spec}, should be in [1, obs_dim={obs_dim}].")
        return np.arange(obs_dim - int(goal_spec), obs_dim, dtype=np.int64)

    goal_indices = np.asarray(goal_spec, dtype=np.int64).reshape(-1)
    if goal_indices.size == 0:
        raise ValueError("task.goal_dim as index list cannot be empty.")

    goal_indices = np.where(goal_indices < 0, goal_indices + obs_dim, goal_indices)
    if np.any(goal_indices < 0) or np.any(goal_indices >= obs_dim):
        raise ValueError(f"task.goal_dim contains out-of-range indices for obs_dim={obs_dim}: {goal_indices.tolist()}")
    if np.unique(goal_indices).size != goal_indices.size:
        raise ValueError(f"task.goal_dim contains duplicate indices: {goal_indices.tolist()}")

    return goal_indices.astype(np.int64)


class Timer:
    def __init__(self):
        self.tik = None

    def start(self):
        self.tik = time.time()

    def stop(self):
        return time.time() - self.tik
    
    
# class Logger:
#     """Primary logger object. Logs in wandb."""
#     def __init__(self, log_dir, cfg):
#         self._log_dir = make_dir(log_dir)
#         self._model_dir = make_dir(self._log_dir / 'models')
#         self._video_dir = make_dir(self._log_dir / 'videos')
#         self._cfg = cfg

#         wandb.init(
#             config=OmegaConf.to_container(cfg),
#             project=cfg.project,
#             group=cfg.group,
#             name=cfg.exp_name,
#             id=str(uuid.uuid4()),
#             mode=cfg.wandb_mode,
#             dir=self._log_dir
#         )
#         self._wandb = wandb

#     def video_init(self, env, enable=False, video_id=""):
#         # assert isinstance(env.env, VideoRecordingWrapper)
#         if isinstance(env.env, VideoRecordingWrapper):
#             video_env = env.env
#         else:
#             video_env = env
#         if enable:
#             video_env.video_recoder.stop()
#             video_filename = os.path.join(self._video_dir, f"{video_id}_{wv.util.generate_id()}.mp4")
#             video_env.file_path = str(video_filename)
#         else:
#             video_env.file_path = None
            
#     def log(self, d, category):
#         assert category in ['train', 'inference']
#         assert 'step' in d
#         print(f"[{d['step']}]", " / ".join(f"{k} {v:.2f}" for k, v in d.items()))
#         with (self._log_dir / "metrics.jsonl").open("a") as f:
#             f.write(json.dumps({"step": d['step'], **d}) + "\n")
#         _d = dict()
#         for k, v in d.items():
#             _d[category + "/" + k] = v
#         self._wandb.log(_d, step=d['step'])
        
#     def save_agent(self, agent=None, identifier='final'):
#         if agent:
#             fp = self._model_dir / f'model_{str(identifier)}.pt'
#         agent.save(fp)
#         print(f"model_{str(identifier)} saved")

#     def finish(self, agent):
#         try:
#             self.save_agent(agent)
#         except Exception as e:
#             print(f"Failed to save model: {e}")
#         if self._wandb:
#             self._wandb.finish()

def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v

def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')
    

def visualize_3d_trajectories_wandb(
    planned_traj: np.ndarray,
    planned_segment_traj: Optional[np.ndarray] = None,
    actual_traj: Optional[np.ndarray] = None,
    n_cols: Optional[int] = None,
    figsize_per_plot: float = 5.0,
) -> wandb.Image:
    """
    可视化 3D 轨迹，并打包成单张网格图，返回 wandb.Image 方便日志记录。
    规划轨迹始终使用 blend 后的 planned_traj，并用虚线绘制。
    实际轨迹使用较浅的实线。marker 语义固定为：
    - 实际起点: 圆形
    - 规划终点: X
    - 实际终点: 方块
    不同 object 使用不同颜色。

    参数:
        planned_traj: numpy.ndarray, shape [B, object_num, T_plan, 3]
            B: batch size
            object_num: 物体数量
            T_plan: 规划序列长度
            末维为 (x, y, z)
        actual_traj: numpy.ndarray, shape [B, object_num, T_actual, 3], 可选
            实际执行轨迹，若为 None 则仅绘制规划轨迹
        n_cols: 网格列数, 如果为 None 则约取 sqrt(B)
        figsize_per_plot: 每个子图的边长（英寸）

    返回:
        wandb.Image
    """
    assert planned_traj.ndim == 4 and planned_traj.shape[-1] == 3, \
        f"planned_traj shape 必须为 [B, object_num, T_plan, 3]，当前为 {planned_traj.shape}"
    if planned_segment_traj is not None:
        assert planned_segment_traj.ndim == 5 and planned_segment_traj.shape[-1] == 3, \
            f"planned_segment_traj shape 必须为 [B, num_segments, object_num, T_seg, 3]，当前为 {planned_segment_traj.shape}"
        assert planned_segment_traj.shape[0] == planned_traj.shape[0] and planned_segment_traj.shape[2] == planned_traj.shape[1], \
            "planned_segment_traj 与 planned_traj 的 B/object_num 维度必须一致"

    if actual_traj is not None:
        assert actual_traj.ndim == 4 and actual_traj.shape[-1] == 3, \
            f"actual_traj shape 必须为 [B, object_num, T_actual, 3]，当前为 {actual_traj.shape}"
        assert actual_traj.shape[0] == planned_traj.shape[0] and actual_traj.shape[1] == planned_traj.shape[1], \
            "planned_traj 与 actual_traj 的 B/object_num 维度必须一致"

    segment_cmap = get_cmap("tab10")
    object_cmap = get_cmap("Dark2")

    B, object_num, _, _ = planned_traj.shape
    use_segment_coloring = planned_segment_traj is not None and planned_segment_traj.shape[1] > 0

    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(B)))
    n_rows = int(math.ceil(B / n_cols))

    fig = plt.figure(figsize=(figsize_per_plot * n_cols, figsize_per_plot * n_rows))
    legend_handles = {}

    def register_legend(handle: Line2D):
        legend_handles.setdefault(handle.get_label(), handle)

    def get_object_color(obj_idx: int):
        return object_cmap(obj_idx % object_cmap.N)

    def lighten_color(color, blend=0.6):
        rgb = np.asarray(color[:3], dtype=np.float32)
        return tuple((1.0 - blend) * rgb + blend * np.ones(3, dtype=np.float32))

    def draw_marker(ax, point, marker, color, size):
        ax.scatter(
            point[0],
            point[1],
            point[2],
            marker=marker,
            s=size,
            c=[color],
            edgecolors='black',
            linewidths=0.7,
            depthshade=False,
            zorder=4,
        )

    register_legend(
        Line2D(
            [0], [0],
            linestyle='None',
            marker='o',
            markersize=7.5,
            markerfacecolor='0.45',
            markeredgecolor='black',
            label="Actual Start",
        )
    )
    register_legend(
        Line2D(
            [0], [0],
            linestyle='None',
            marker='X',
            markersize=8.5,
            markerfacecolor='0.45',
            markeredgecolor='0.45',
            label="Planned Goal",
        )
    )
    register_legend(
        Line2D(
            [0], [0],
            linestyle='None',
            marker='s',
            markersize=7.5,
            markerfacecolor='0.45',
            markeredgecolor='black',
            label="Actual End",
        )
    )
    if not use_segment_coloring:
        for o in range(object_num):
            obj_color = get_object_color(o)
            register_legend(
                Line2D(
                    [0], [0],
                    linestyle='None',
                    marker='o',
                    markersize=8,
                    markerfacecolor=obj_color,
                    markeredgecolor='black',
                    label=f"Object {o + 1}",
                )
            )
    if actual_traj is not None:
        register_legend(
            Line2D([0], [0], color='0.65', linestyle='-', linewidth=1.8, label="Actual Trajectory")
        )
    if use_segment_coloring:
        n_segments = int(planned_segment_traj.shape[1])
        for s in range(n_segments):
            seg_color = segment_cmap(s % segment_cmap.N)
            register_legend(
                Line2D([0], [0], color=seg_color, linestyle='--', linewidth=2.1, label=f"Segment {s + 1}")
            )

    for b in range(B):
        ax = fig.add_subplot(n_rows, n_cols, b + 1, projection="3d")
        ax.set_title(f"Sample {b}", fontsize=8)

        all_xyz = [planned_traj[b].reshape(-1, 3)]
        if planned_segment_traj is not None:
            all_xyz.append(planned_segment_traj[b].reshape(-1, 3))
        if actual_traj is not None:
            all_xyz.append(actual_traj[b].reshape(-1, 3))
        all_xyz = np.concatenate(all_xyz, axis=0)
        finite_mask = np.isfinite(all_xyz).all(axis=1)
        all_xyz = all_xyz[finite_mask] if finite_mask.any() else np.zeros((1, 3), dtype=np.float32)

        if actual_traj is not None:
            for o in range(object_num):
                obj_color = get_object_color(o)
                real_color = lighten_color(obj_color, blend=0.58)
                real_o = actual_traj[b, o]
                real_mask = np.isfinite(real_o).all(axis=1)
                real_o = real_o[real_mask]
                if real_o.shape[0] == 0:
                    continue
                ax.plot(
                    real_o[:, 0],
                    real_o[:, 1],
                    real_o[:, 2],
                    linestyle='-',
                    linewidth=1.9,
                    color=real_color,
                    alpha=0.9,
                    zorder=1,
                )

        if use_segment_coloring:
            n_segments = int(planned_segment_traj.shape[1])
            seg_horizon = int(planned_segment_traj.shape[3])
            plan_horizon = int(planned_traj.shape[2])
            if n_segments <= 1:
                segment_stride = seg_horizon
            else:
                segment_stride = max(1, int(round((plan_horizon - seg_horizon) / float(n_segments - 1))))

            for s in range(n_segments):
                seg_color = segment_cmap(s % segment_cmap.N)
                seg_start = s * segment_stride
                seg_end = min(seg_start + seg_horizon, plan_horizon)
                if seg_start >= seg_end:
                    continue

                for o in range(object_num):
                    plan_o = planned_traj[b, o]
                    seg_o = plan_o[seg_start:seg_end]
                    seg_mask = np.isfinite(seg_o).all(axis=1)
                    seg_o = seg_o[seg_mask]
                    if seg_o.shape[0] == 0:
                        continue
                    ax.plot(
                        seg_o[:, 0],
                        seg_o[:, 1],
                        seg_o[:, 2],
                        linestyle='--',
                        linewidth=2.1,
                        color=seg_color,
                        alpha=0.95,
                        zorder=2,
                    )
                    ax.scatter(
                        seg_o[:, 0],
                        seg_o[:, 1],
                        seg_o[:, 2],
                        s=10,
                        c=[seg_color],
                        alpha=0.5,
                        depthshade=False,
                        zorder=3,
                    )
        else:
            for o in range(object_num):
                plan_color = get_object_color(o)
                plan_o = planned_traj[b, o]
                plan_mask = np.isfinite(plan_o).all(axis=1)
                plan_o = plan_o[plan_mask]
                if plan_o.shape[0] == 0:
                    continue
                ax.plot(
                    plan_o[:, 0],
                    plan_o[:, 1],
                    plan_o[:, 2],
                    linestyle='--',
                    linewidth=2.1,
                    color=plan_color,
                    alpha=0.95,
                    zorder=2,
                )
                ax.scatter(
                    plan_o[:, 0],
                    plan_o[:, 1],
                    plan_o[:, 2],
                    s=10,
                    c=[plan_color],
                    alpha=0.5,
                    depthshade=False,
                    zorder=3,
                )

                register_legend(
                    Line2D([0], [0], color=plan_color, linestyle='--', linewidth=2.1, label=f"Object {o + 1} Planned")
                )

        for o in range(object_num):
            obj_color = get_object_color(o)

            plan_o = planned_traj[b, o]
            plan_mask = np.isfinite(plan_o).all(axis=1)
            plan_o = plan_o[plan_mask]
            if plan_o.shape[0] > 0:
                draw_marker(ax, plan_o[-1], 'X', obj_color, 80)

            if actual_traj is not None:
                real_o = actual_traj[b, o]
                real_mask = np.isfinite(real_o).all(axis=1)
                real_o = real_o[real_mask]
                if real_o.shape[0] > 0:
                    draw_marker(ax, real_o[0], 'o', obj_color, 58)
                    draw_marker(ax, real_o[-1], 's', obj_color, 62)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 适当设置相同尺度，避免比例失真
        x_min, y_min, z_min = all_xyz.min(axis=0)
        x_max, y_max, z_max = all_xyz.max(axis=0)
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) + 1e-6
        x_center = (x_max + x_min) / 2.0
        y_center = (y_max + y_min) / 2.0
        z_center = (z_max + z_min) / 2.0
        half = max_range / 2.0
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_zlim(z_center - half, z_center + half)

    if legend_handles:
        handles = list(legend_handles.values())
        fig.legend(handles=handles, loc='upper center', ncol=max(1, min(len(handles), 5)), fontsize=8.5)
        plt.tight_layout(rect=(0, 0, 1, 0.88))
    else:
        plt.tight_layout()

    # 转成 numpy 图像 (H, W, C) 用于 wandb.Image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8) # (H, W, 4)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (H, W, 4)
    img = buf[:, :, :3]
    plt.close(fig)

    return wandb.Image(img)

    
def save_3d_trajectories_image(
    traj: np.ndarray,
    path: str,
    n_cols: Optional[int] = None,
    cmap_name: str = "viridis",
    figsize_per_plot: float = 4.0,
):
    """
    可视化 3D 轨迹，并直接保存到本地图片文件。

    参数:
        traj: numpy.ndarray, shape [B, object_num, T, 3]
        path: 输出图片路径，例如 "traj_3d_test.png"
    """
    assert traj.ndim == 4 and traj.shape[-1] == 3, \
        f"traj shape 必须为 [B, object_num, T, 3]，当前为 {traj.shape}"

    B, object_num, T, _ = traj.shape

    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(B)))
    n_rows = int(math.ceil(B / n_cols))

    fig = plt.figure(figsize=(figsize_per_plot * n_cols, figsize_per_plot * n_rows))
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=T - 1)

    for b in range(B):
        ax = fig.add_subplot(n_rows, n_cols, b + 1, projection="3d")
        ax.set_title(f"Sample {b}", fontsize=8)

        for o in range(object_num):
            traj_o = traj[b, o]  # [T, 3]
            xs, ys, zs = traj_o[:, 0], traj_o[:, 1], traj_o[:, 2]

            colors = cmap(norm(np.arange(T)))
            ax.scatter(xs, ys, zs, c=colors, s=5)

            ax.scatter(xs[0], ys[0], zs[0],
                       c=[cmap(norm(0))], marker="+", s=50, linewidths=2)
            ax.scatter(xs[-1], ys[-1], zs[-1],
                       c=[cmap(norm(T - 1))], marker="x", s=50, linewidths=2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        all_xyz = traj[b].reshape(-1, 3)
        x_min, y_min, z_min = all_xyz.min(axis=0)
        x_max, y_max, z_max = all_xyz.max(axis=0)
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) + 1e-6
        x_center = (x_max + x_min) / 2.0
        y_center = (y_max + y_min) / 2.0
        z_center = (z_max + z_min) / 2.0
        half = max_range / 2.0
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_zlim(z_center - half, z_center + half)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
