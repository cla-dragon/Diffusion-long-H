
import torch
import torch.nn as nn
from typing import Optional

# 模拟一个简单的 Diffusion Backbone
from cleandiffuser.nn_diffusion import BaseNNDiffusion

class DummyDiffusionNN(BaseNNDiffusion):
    def __init__(self, x_shape):
        super().__init__(emb_dim=64)
        self.x_shape = x_shape
        # 一个简单的 MLP，输入 (B, C, L) 或 (B, L, C) -> 输出同维度
        # 假设输入是 (B, L, C)
        self.net = nn.Sequential(
            nn.Linear(x_shape[-1], 64),
            nn.ReLU(),
            nn.Linear(64, x_shape[-1])
        )

    def forward(self, x, t, condition=None, mask=None):
        # x: (B, L, C)
        # t: (B,)
        # 简单地返回一个相关的输出，这里为了测试甚至可以返回随机值或 0
        # 为了让输出有点意义，我们让它预测噪声，这里随便预测一点
        return self.net(x)

# 导入我们需要测试的类
from cleandiffuser_sup.diffusion.repaint_sde import RepaintContinuousDiffusionSDE

def test_repaint_consistency():
    # 1. 设置参数
    bs = 2
    seq_len = 20
    state_dim = 4
    x_shape = (seq_len, state_dim)
    device = "cpu"

    print(f"Testing Repaint Consistency with Input Shape: [Batch={bs}, Seq={seq_len}, Dim={state_dim}]")

    # 2. 准备模型
    nn_diffusion = DummyDiffusionNN(x_shape).to(device)
    
    # 实例化 RepaintModel
    # 注意：训练时通常不传 fix_mask，或者传 None
    model = RepaintContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion,
        device=device,
        fix_mask=None,  # 确保不受基础类 fix_mask 逻辑影响
        noise_schedule="linear", # 使用简单的线性调度
    )

    # 3. 构造 Prior 数据 (模拟真实的轨迹)
    # 假设是一条直线轨迹
    prior = torch.zeros((bs, seq_len, state_dim), device=device)
    for i in range(seq_len):
        prior[:, i, :] = i / seq_len  # 简单的线性增加

    # 4. 构造 Mask
    # 我们只固定起点 (t=0) 和 终点 (t=seq_len-1)
    mask = torch.zeros((bs, seq_len, state_dim), device=device)
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0
    
    print("\n[Mask Setup]")
    print(f"Mask shape: {mask.shape}")
    print(f"Fixed indices: 0 and {seq_len-1}")

    # 5. 运行推理
    print("\nStarting Sampling (Repaint)...")
    
    # 增加采样步数以配合 jump_len
    sample_steps = 20
    repaint_times = 5
    jump_len = 5 # 测试一下 Jump 逻辑
    
    print(f"Configs: sample_steps={sample_steps}, repaint_times={repaint_times}, jump_len={jump_len}")

    # 注意：Prior 必须传入。对于非 masked 区域，prior 的值其实不重要，
    # 但对于 masked 区域，prior 的值就是由 mask 指定的已知值。
    x_gen, log = model.sample(
        prior=prior,
        mask=mask,
        repaint_times=repaint_times,
        jump_len=jump_len,
        sample_steps=sample_steps,
        solver="ddpm", # 使用 DDPM 求解器
        n_samples=bs
    )

    print("Sampling Finished.")

    # 6. 验证结果
    # 提取被 Mask 固定的部分
    # x_gen: (B, L, C)
    
    # 检查起点
    prior_start = prior[:, 0, :]
    gen_start = x_gen[:, 0, :]
    diff_start = (prior_start - gen_start).abs().mean()
    
    # 检查终点
    prior_end = prior[:, -1, :]
    gen_end = x_gen[:, -1, :]
    diff_end = (prior_end - gen_end).abs().mean()
    
    # 检查中间（应该是生成的，不一定等于 prior，因为 prior 这里是虚构的直线，生成模型是随机初始化的 Dummy）
    # 但我们只关心 Mask 部分是否贴合
    
    print("\n[Consistency Check]")
    print(f"Start (t=0) Mean L1 Error: {diff_start.item():.6f}")
    print(f"End (t={seq_len-1}) Mean L1 Error: {diff_end.item():.6f}")

    threshold = 1e-2
    if diff_start < threshold and diff_end < threshold:
        print("\n✅ Test Passed: Generated sequence matches prior at masked positions.")
    else:
        print("\n❌ Test Failed: Significant deviation at masked positions.")
        print("Note: Small errors are expected due to numerical precision and noise at the final step (sigma_0 > 0).")
        print("If errors are huge, the mask mixing logic might be wrong.")

if __name__ == "__main__":
    test_repaint_consistency()
