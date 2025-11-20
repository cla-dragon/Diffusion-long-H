import random
import time
import uuid
import os
import json
import wandb
import wandb.sdk.data_types.video as wv
import numpy as np
import torch
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
    
    


    
