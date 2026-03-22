import argparse
import random

import numpy as np
import torch

from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser_sup.diffusion import CurvatureContinuousDiffusionSDE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_id_batch(batch_size: int, horizon: int, obs_dim: int, device: torch.device) -> torch.Tensor:
    start = torch.randn((batch_size, obs_dim), device=device)
    goal = torch.randn((batch_size, obs_dim), device=device)
    alphas = torch.linspace(0.0, 1.0, horizon, device=device).view(1, horizon, 1)
    traj = (1.0 - alphas) * start.unsqueeze(1) + alphas * goal.unsqueeze(1)
    traj = traj + 0.05 * torch.randn_like(traj)
    return traj


def make_priors(n: int, horizon: int, obs_dim: int, device: torch.device, ood: bool = False) -> torch.Tensor:
    start = torch.randn((n, obs_dim), device=device)
    if ood:
        goal = 4.0 + 1.5 * torch.randn((n, obs_dim), device=device)
    else:
        goal = torch.randn((n, obs_dim), device=device)

    # Build smooth trajectory-shaped priors: linear path + a single smooth bend.
    alphas = torch.linspace(0.0, 1.0, horizon, device=device).view(1, horizon, 1)
    base = (1.0 - alphas) * start.unsqueeze(1) + alphas * goal.unsqueeze(1)

    direction = goal - start
    direction_norm = torch.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
    direction_unit = direction / direction_norm

    bend_dir = torch.randn((n, obs_dim), device=device)
    bend_dir = bend_dir - (bend_dir * direction_unit).sum(dim=-1, keepdim=True) * direction_unit
    bend_norm = torch.norm(bend_dir, dim=-1, keepdim=True)
    fallback = torch.randn_like(bend_dir)
    bend_dir = torch.where(bend_norm > 1e-6, bend_dir / bend_norm.clamp_min(1e-6), fallback)

    bend_scale = (0.08 + 0.10 * torch.rand((n, 1), device=device)) * direction_norm
    bend_profile = torch.sin(torch.pi * alphas)
    prior = base + bend_profile * bend_scale.unsqueeze(1) * bend_dir.unsqueeze(1)

    prior = prior + 0.01 * torch.randn_like(prior)
    prior[:, 0, :] = start
    prior[:, -1, :] = goal
    return prior


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main():
    parser = argparse.ArgumentParser(description="Curvature-based OOD test for trajectory diffusion.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--n-eval", type=int, default=128)
    parser.add_argument("--solver", type=str, default="ddim")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    nn_diffusion = DiT1d(
        in_dim=args.obs_dim,
        emb_dim=64,
        d_model=128,
        n_heads=4,
        depth=3,
        timestep_emb_type="fourier",
    )

    fix_mask = torch.zeros((args.horizon, args.obs_dim), device=device)
    fix_mask[0] = 1.0
    fix_mask[-1] = 1.0

    loss_weight = torch.ones((args.horizon, args.obs_dim), device=device)

    planner = CurvatureContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=None,
        device=device,
        predict_noise=True,
        noise_schedule="linear",
    )

    planner.train()
    for step in range(1, args.train_steps + 1):
        batch = make_id_batch(args.batch_size, args.horizon, args.obs_dim, device)
        log = planner.update(batch)
        if step % 100 == 0:
            print(f"[train] step={step:4d} loss={log['loss']:.6f}")

    planner.eval()
    with torch.no_grad():
        prior_id = make_priors(args.n_eval, args.horizon, args.obs_dim, device, ood=False)
        prior_ood = make_priors(args.n_eval, args.horizon, args.obs_dim, device, ood=True)

    _, log_id = planner.sample(
        prior=prior_id,
        solver=args.solver,
        n_samples=args.n_eval,
        sample_steps=args.sample_steps,
        use_ema=True,
        temperature=1.0,
        return_curvature_per_step=False,
    )
    _, log_ood = planner.sample(
        prior=prior_ood,
        solver=args.solver,
        n_samples=args.n_eval,
        sample_steps=args.sample_steps,
        use_ema=True,
        temperature=1.0,
        return_curvature_per_step=False,
    )

    curvature_id = np.asarray(log_id["curvature"], dtype=np.float64)
    curvature_ood = np.asarray(log_ood["curvature"], dtype=np.float64)

    labels = np.concatenate([
        np.zeros_like(curvature_id, dtype=np.int64),
        np.ones_like(curvature_ood, dtype=np.int64),
    ])
    scores = np.concatenate([curvature_id, curvature_ood])
    auc = binary_auc(labels, scores)

    print("\n===== Curvature OOD Test =====")
    print(f"ID mean curvature   : {curvature_id.mean():.6f} +/- {curvature_id.std():.6f}")
    print(f"OOD mean curvature  : {curvature_ood.mean():.6f} +/- {curvature_ood.std():.6f}")
    print(f"AUROC(curvature->OOD): {auc:.6f}")
    print("Top-10 high-curvature scores:")
    print(np.sort(scores)[-10:])

if __name__ == "__main__":
    main()
