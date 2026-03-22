# CDGS GSC-Style Mechanism Mapping to Diffusion-long-H

This note records the detailed GSC-related mechanisms from CDGS_ogbench and maps them to the current Diffusion-long-H implementation, so future development can continue in this repository directly.

## Scope and source files in CDGS_ogbench

Key files inspected:

- diffuser/models/cd_stgl_sml_dfu/stgl_sml_policy_v1.py
- diffuser/models/cd_stgl_sml_dfu/stgl_sml_diffusion_v1.py
- diffuser/guides/comp/traj_blender.py
- diffuser/utils/cp_utils/cp_luo_utils.py

## CDGS GSC mechanism details

1. Overlap-distance ranking across candidate trajectories
- CDGS computes overlap mismatch between adjacent chunks on each candidate sample:
  - extract previous chunk tail overlap and next chunk head overlap
  - compute mean squared distance
  - sum across chunk pairs to get per-sample score
- Relevant functions:
  - compute_ovlp_dist(...) in cp_luo_utils.py
  - Called in stgl_sml_policy_v1.py before candidate selection

2. Keep top_n candidates
- Candidates are sorted by overlap-distance ascending (smaller is better)
- Keep top_n candidates only
- Relevant function:
  - pick_top_n_trajs(...) in cp_luo_utils.py

3. Blender fusion for overlap windows
- CDGS fuses overlap region with schedule-based weights (not only simple average)
- Supported blend types:
  - exp / cosine / linear / smoothstep
- Implemented in:
  - blend_2_np_trajs_23d(...) in traj_blender.py
  - Traj_Blender.blend_traj_lists(...)

4. GSC internal late-stage feasibility filtering (inside denoising)
- In comp_pred_p_loop_n_GSC(...), CDGS computes inversion score near late denoising steps
- Keeps top 10 percent and repopulates batch from selected indices
- Stores per-component feasibility scores in last_gsc_comp_scores

5. Training-time mixed conditioning on boundaries
- For each side, CDGS samples overlap-conditioning or inpaint-conditioning by probability
- Start/end side have independent behavior
- Enforces side exclusivity and optional full-drop logic

## What has been implemented in Diffusion-long-H

Implemented files:

- cleandiffuser_sup/diffusion/multiseg_sde.py
- evaluate/multiseg.py
- configs/diffuser_test/ogbench/ogbench_multiseg.yaml

### Implemented mechanisms

1. Training mixed boundary conditioning
- In MultiSegmentRepaintDiffusionSDE.loss(...):
  - overlap conditioning on left/right boundaries by overlap_prob
  - inpaint conditioning on start/end by inpaint_start_prob and inpaint_end_prob
  - inpaint anchors are excluded from denoising loss

2. Multi-segment generation with iterative overlap consistency updates
- In sample_multi_segment(...):
  - first segment hard-constrained to start
  - last segment hard-constrained to goal
  - middle segments constrained by neighbor overlaps
  - inner resample loop performs repeated repaint denoising and overlap fusion

3. Overlap blender fusion
- Added configurable overlap_blend_type with:
  - avg / exp / cosine / linear / smoothstep
- Added overlap_exp_beta for exponential blending sharpness

4. Overlap-distance candidate scoring and ranking
- sample_multi_segment(...) now computes candidate_overlap_cost
- evaluate/multiseg.py ranks candidates by overlap cost ascending

5. Keep top_n and pick policy
- evaluate/multiseg.py keeps top_n_candidates
- pick_mode supports:
  - first
  - rand
  - score_weighted (soft preference by overlap score)

## Mechanisms not yet implemented in Diffusion-long-H

1. CDGS-style late-stage inversion-score filtering inside denoising loop
- Current implementation ranks by overlap mismatch after segment updates
- Not yet doing the in-loop top-10-percent repopulation by inversion score

2. CDGS per-component feasibility trace export
- last_gsc_comp_scores equivalent is not yet exposed
- Only aggregate candidate overlap cost is currently logged

3. Full CDGS policy object and rich planning outputs
- Current path keeps Diffusion-long-H style evaluate pipeline
- Does not port CDGS-specific policy wrapper class and all its rich debug outputs

## Recommended next extension order

1. Add optional inversion-feasibility filter in sample_multi_segment(...)
- Late denoise window
- Keep ratio configurable
- Repopulate selected candidates

2. Add per-boundary diagnostic logs
- overlap cost per boundary pair
- not only total candidate score

3. Add evaluate-time debug table output
- chosen index
- top_n scores
- pick mode and probabilities when score_weighted

## Minimal config knobs for current implementation

- multi_segment.overlap_blend_type
- multi_segment.overlap_exp_beta
- multi_segment.top_n_candidates
- multi_segment.pick_mode
- multi_segment.score_temperature
- training_conditioning.overlap_prob
- training_conditioning.inpaint_start_prob
- training_conditioning.inpaint_end_prob
