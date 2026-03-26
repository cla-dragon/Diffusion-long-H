# Segment Stitch Planner for Diffusion-long-H

这个 README 记录本次新增的长时域多段 diffusion 路径，以及各文件在仓库里的职责。

## 新增内容

- `CDGS_Long_Horizon_Algorithm.md`
  - 先解释 CDGS 原始训练与采样机制，再给出本仓库里的等价约束。

- `cleandiffuser_sup/nn_condition/segment_boundary.py`
  - 新的边界条件编码器。
  - 把左右 overlap 片段、边界模式、以及对应噪声层级编码成全局条件向量。

- `cleandiffuser_sup/diffusion/segment_stitch_discrete.py`
  - 新的离散扩散训练类。
  - 训练阶段支持 overlap / inpaint / side-drop 混合边界条件。
  - 推理阶段提供 `interleave` 与 `gsc` 两种多段生成路径。

- `evaluate/segment_stitch.py`
  - 新的长轨迹评测函数。
  - 负责候选排序、overlap 融合、子目标推进和环境执行。

- `pipelines/segment_stitch_diffuser_ogbench.py`
  - OGBench 训练 / 推理入口。

- `configs/diffuser_test/ogbench/ogbench_segment_stitch.yaml`
  - 默认配置文件。

## 实现逻辑

### 训练

每条短段轨迹都在训练时随机采样左右边界模式：

- overlap
- inpaint
- drop

其中：

- overlap 会把真实边界片段按当前扩散层级加噪后作为条件输入。
- inpaint 会把首 token 或尾 token 强行钉到真实状态。
- 被 inpaint 的位置不参与去噪 loss。

### 推理一：Interleave

- 每个扩散步内按段从左到右更新。
- 中间段左边看到的是刚更新过的上一段尾部，右边看到的是尚未更新的下一段头部。
- 这正是 CDGS 原始 interleave 的交错传播方式。

### 推理二：GSC

- 每个扩散步内部先强制平均所有相邻 overlap。
- 然后做一次去噪。
- 如果还没完成这一扩散步的内循环，就把结果再加噪回当前层级。
- 在采样尾段再做 inversion-based feasibility filtering，只保留最稳定的一部分样本继续传播。

### 候选挑选

- 先在未融合前的段序列上计算 overlap mismatch。
- 按 mismatch 排序。
- 只取 `top_n`。
- 再用 `exp / cosine / linear / smoothstep / avg` 之一对 overlap 做最终融合，得到完整长轨迹。

## 使用方式

训练：

```bash
python pipelines/segment_stitch_diffuser_ogbench.py mode=train
```

推理：

```bash
python pipelines/segment_stitch_diffuser_ogbench.py mode=inference planner_ckpt=latest
```

切到 GSC-style：

```bash
python pipelines/segment_stitch_diffuser_ogbench.py stitch.strategy=gsc mode=inference
```

离线数据集训练 + 可视化 demo：

```bash
conda run -n ogbench python pipelines/demo_segment_stitch_offline.py \
  --mode train_and_visualize \
  --device cpu \
  --total-horizon 48 \
  --num-segments 2 \
  --overlap-steps 2 \
  --output-path results/multiseg_diffuser/cube_single_offline_demo_h48.png
```

这个 demo 不跑真实环境，只从 `cube-single-play-v0` 数据集里截一个总长为 `48` 的真实窗口，取其 start / goal，先训练一个适配多段拼接的 planner，再分别生成 `interleave` 和 `gsc` 两条拼接轨迹，并画出 object 的 `xyz`。

训练得到的多段 diffuser 默认保存在：

```text
results/multiseg_diffuser/cube-single-play-v0_Multi_SegH25/demo_multiseg_planner_ckpt_latest.pt
```

如果只想复用已经训练好的模型做可视化：

```bash
conda run -n ogbench python pipelines/demo_segment_stitch_offline.py \
  --mode visualize \
  --device cpu \
  --total-horizon 48 \
  --num-segments 2 \
  --overlap-steps 2 \
  --segment-stitch-checkpoint results/multiseg_diffuser/cube-single-play-v0_Multi_SegH25/demo_multiseg_planner_ckpt_latest.pt \
  --output-path results/multiseg_diffuser/cube_single_offline_demo_h48.png
```

## 关键配置

- `task.planner_horizon`
  - 目标长轨迹长度，沿用仓库原有任务配置，单位是环境步数。

- `stitch.segment_model_horizon`
  - 单段 diffusion 模型看到的短轨迹长度，单位是 downsample 后的模型步数。

- `stitch.overlap_steps`
  - 相邻短段之间的重叠长度，单位也是模型步数。

- `training_boundary.overlap_prob`
  - 每侧使用 overlap 条件的概率。

- `training_boundary.side_drop_prob`
  - 每侧完全 drop 的概率。

- `stitch.condition_guidance_scale`
  - interleave 里 overlap 条件的 guidance 强度。

- `stitch.gsc_inner_loops`
  - GSC 每个扩散步内部的平均-重采样次数。

- `stitch.gsc_keep_ratio`
  - GSC late-stage feasibility filtering 保留比例。

## 设计取向

本次实现刻意保持“简单明了”：

- 不把 CDGS 的命名直接搬进来。
- 不重写现有单段 planner。
- 通过新的离散 planner 路径把核心算法单独收拢起来，避免和现有 continuous planner 互相污染。
