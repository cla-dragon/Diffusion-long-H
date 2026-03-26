# CDGS Long-Horizon Composition in Diffusion-long-H

本文件先把 `CDGS_ogbench` 中与长时域多段扩散直接相关的机制拆开，再说明本仓库里的等价实现约束。

## 1. 参考源码范围

本次只以 `CDGS_ogbench` 真实源码为依据，核心文件是：

- `diffuser/models/cd_stgl_sml_dfu/stgl_sml_diffusion_v1.py`
- `diffuser/models/cd_stgl_sml_dfu/stgl_sml_policy_v1.py`
- `diffuser/models/cd_stgl_sml_dfu/stgl_sml_temporal_cond_v1.py`
- `diffuser/guides/comp/traj_blender.py`
- `diffuser/utils/cp_utils/cp_luo_utils.py`

## 2. 训练机制

CDGS 不是只训练“起点 inpaint + 终点 inpaint”的普通 goal-conditioned diffusion，而是显式训练模型同时具备两类边界能力：

1. `inpaint` 能力
- 左端可以被固定到真实起点。
- 右端可以被固定到真实终点。
- 被固定的位置在训练 loss 中不再承担去噪回归。

2. `overlap` 能力
- 左端可以接收“来自前一段尾部”的重叠窗口。
- 右端可以接收“来自后一段头部”的重叠窗口。
- 这些重叠窗口不是直接拼到目标轨迹里，而是作为边界条件输入给去噪网络。

3. 左右两侧独立采样
- 每个样本的左侧、右侧分别独立决定使用 `overlap` 还是 `inpaint`。
- 每侧还存在独立的“全丢弃”概率，用来让模型见到无边界条件样本。
- 因此训练分布覆盖：
  - 左 overlap / 右 overlap
  - 左 overlap / 右 inpaint
  - 左 inpaint / 右 overlap
  - 左 inpaint / 右 inpaint
  - 单侧或双侧 drop

4. overlap 条件与当前样本噪声层级匹配
- CDGS 会给左右 overlap 片段分别加噪。
- 加噪层级与主样本扩散层级相同，或者减一层。
- 这个细节的作用是避免边界条件与主体状态的噪声层级严重错位。

5. 明确区分边界模式
- 模型不会只靠“输入值长什么样”来猜当前侧是 overlap 还是 inpaint。
- 源码里还显式给每侧注入模式标记，让网络知道：
  - 这一侧是 overlap
  - 这一侧是 inpaint
  - 这一侧被 drop

## 3. 推理机制一：Interleave

`interleave` 是 CDGS 的原始组合方式。设短段长度为 `H`，重叠长度为 `L`，共有 `K` 段。

### 3.1 外层扩散循环

对每个扩散步，从高噪声到低噪声进行迭代。

### 3.2 段内更新顺序

同一个扩散步里，按段顺序从左到右更新：

1. 第一段
- 左端做起点 inpaint。
- 右端拿第二段当前噪声层级的前 `L` 个状态作为 overlap 条件。

2. 中间段
- 左端拿前一段“刚更新过”的尾部 overlap。
- 右端拿后一段“尚未更新”的头部 overlap。
- 也就是说左边界更干净，右边界更噪。

3. 最后一段
- 左端拿前一段刚更新过的尾部 overlap。
- 右端做终点 inpaint。

### 3.3 Interleave 的本质

Interleave 的核心不是强行平均重叠段，而是：

- 让每一段在去噪时都显式看到邻段边界；
- 通过“左已更新、右未更新”的交错次序，把段间信息逐步往整条轨迹传播。

## 4. 推理机制二：GSC-style

`GSC-style` 在 CDGS 里不是简单换个名字，而是换了一种组合策略。

### 4.1 每个扩散步内部再做内循环

在单个扩散步内部，重复若干次：

1. 先把所有相邻段的 overlap 区域做强制平均。
2. 再对每一段做一次去噪。
3. 如果还没到这一扩散步的最后一次内循环，就把结果重新加噪回当前层级。

这就是“平均 -> 去噪 -> 回跳 -> 再平均”的 resample 结构。

### 4.2 GSC 去噪时的条件形式

GSC 与 interleave 的一个关键差别是：

- 中间段在去噪时不再显式接收 overlap 条件。
- 中间段主要依靠前一步强制平均后的当前轨迹本身来继续收敛。
- 第一段只保留起点 inpaint。
- 最后一段只保留终点 inpaint。

因此 GSC 的 overlap 约束主要来自“当前状态的硬平均”和“内循环反复重采样”，而不是来自去噪网络内部的双侧 overlap 条件编码。

### 4.3 Late-stage feasibility filtering

当采样进入最后一小段扩散步时，CDGS 会：

1. 对每一段计算一个 inversion-based stability score。
2. 将所有段的 score 按样本求平均。
3. 只保留分数最好的前 10% 样本。
4. 再把这部分样本重复扩充回原始 batch 大小，继续后续去噪。

这个步骤只属于 GSC-style，不属于 interleave。

它的作用是：

- 在采样快结束时剔除“不稳定”的候选；
- 让 batch 预算更集中地服务于较可行的组合轨迹。

## 5. 候选排序与最终拼接

无论底层使用 interleave 还是 GSC-style，CDGS 的 policy 侧都会再做一次候选排序：

1. 对每个候选样本，取所有相邻段的 overlap 区域。
2. 计算尾段 overlap 与下一段头 overlap 的均方误差。
3. 将所有相邻段的误差求和，得到该候选的总 overlap mismatch。
4. 按 mismatch 从小到大排序。
5. 只保留 `top_n` 候选。
6. 最终把保留下来的段序列通过 overlap blender 融成完整长轨迹。

这里的 blender 支持：

- `exp`
- `cosine`
- `linear`
- `smoothstep`

如果只想严格复现“生成过程中强制平均”的约束，也可以选 `avg`。

## 6. 在 Diffusion-long-H 中的等价实现要求

为保持算法一致，本仓库里的新实现必须满足：

1. 训练时同时覆盖 overlap / inpaint 两种边界模式。
2. 左右两侧独立采样边界模式，并允许 side-drop。
3. Interleave 采样按“左先更新、右后更新”的交错顺序构造条件。
4. GSC-style 采样按“平均 overlap -> 去噪 -> 回跳 resample”的内循环执行。
5. GSC-style 保留 late-stage feasibility filter。
6. 评测时先按 overlap mismatch 排序，再做最终 overlap 融合与候选挑选。

## 7. 本仓库里的实现映射

新的实现路径拆成四块：

1. 新条件编码器
- 负责把左右 overlap 片段、对应噪声层级、以及左右边界模式编码成全局条件向量。

2. 新离散 diffusion 类
- 训练阶段实现 mixed boundary conditioning。
- 推理阶段实现 interleave 与 GSC-style 两套多段采样循环。

3. 新评测入口
- 负责长轨迹候选排序、overlap 融合、子目标选取与环境执行。

4. 新 pipeline / config / README
- 让这套长时域多段扩散路径与当前仓库已有 OGBench 训练流程并行存在。
