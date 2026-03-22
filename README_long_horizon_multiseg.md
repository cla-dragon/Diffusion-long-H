# Long-Horizon Multi-Segment Diffusion (Diffusion-long-H)

This document describes the newly added long-horizon trajectory generation path built in the style of this repository.

## What is added

- New planner class: `cleandiffuser_sup/diffusion/multiseg_sde.py`
  - `MultiSegmentRepaintDiffusionSDE`
- New evaluation entry: `evaluate/multiseg.py`
  - `multi_segment_evaluate(...)`
- New pipeline entry: `pipelines/multiseg_diffuser_ogbench.py`
- New config: `configs/diffuser_test/ogbench/ogbench_multiseg.yaml`

## Core idea

The planner combines two conditioning mechanisms:

1. Inpaint endpoint conditioning
- First segment is constrained to start from the current state.
- Last segment is constrained to end at the goal state.

2. Overlap chunk conditioning
- Neighboring segments share an overlap window.
- During generation, overlap chunks are hard-averaged.
- Inner resample loops repeatedly denoise with overlap constraints, then re-average.

This gives a practical long-horizon stitching behavior while keeping code close to existing RePaint diffusion usage.

## Training behavior

`MultiSegmentRepaintDiffusionSDE.loss(...)` uses mixed boundary-conditioning modes per batch sample:

- overlap mode (left/right side independently sampled)
- inpaint mode (start/end token independently sampled)

Configurable probabilities:

- `training_conditioning.overlap_prob`
- `training_conditioning.inpaint_start_prob`
- `training_conditioning.inpaint_end_prob`

For strict inpaint anchors, denoising loss at the anchor tokens is masked out.

## Inference behavior

`sample_multi_segment(...)`:

1. Split plan into multiple segments.
2. Apply fixed conditions:
- segment 0 token 0 = start
- last segment token -1 = goal
3. Use adjacent segment overlap windows as hard known chunks.
4. Repeat inner loops:
- repaint denoise each segment
- fuse overlap windows with configurable blender weights
5. Concatenate segments by removing duplicate overlap windows.

In addition, candidate trajectories are ranked by overlap disagreement score,
then top-n candidates are retained for policy pick.

## How to run

Example (train):

```bash
python pipelines/multiseg_diffuser_ogbench.py mode=train
```

Example (inference):

```bash
python pipelines/multiseg_diffuser_ogbench.py mode=inference planner_ckpt=latest
```

## Main knobs

- Total plan horizon: `task.planner_horizon` (+ `task.jump_steps`)
- Segment size: `multi_segment.segment_horizon`
- Overlap size: `multi_segment.overlap_length`
- Number of segments: `multi_segment.num_segments` (`<=0` means auto)
- Inner consistency loop: `multi_segment.inner_resample_rounds`
- RePaint controls: `multi_segment.repaint_times`, `multi_segment.jump_len`
- Blender controls: `multi_segment.overlap_blend_type`, `multi_segment.overlap_exp_beta`
- Candidate selection: `multi_segment.top_n_candidates`, `multi_segment.pick_mode`, `multi_segment.score_temperature`

## Notes

- This path is added in parallel and does not replace existing single-layer planning.
- The implementation intentionally stays simple and explicit for easier debugging and extension.
