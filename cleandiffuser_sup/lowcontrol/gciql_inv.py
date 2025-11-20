from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent, Normal


class GoalConditionedValue(nn.Module):
    """V(s, g)."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hidden_dims: Iterable[int],
        layer_norm: bool,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        in_dim = obs_dim + goal_dim

        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            last_dim = h

        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(last_dim, 1)

    def forward(self, obs: Tensor, goal: Tensor) -> Tensor:
        x = torch.cat([obs, goal], dim=-1)
        features = self.body(x)
        return self.head(features)


class GoalConditionedCritic(nn.Module):
    """Twin Q(s, a, g)."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int],
        layer_norm: bool,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        in_dim = obs_dim + goal_dim + action_dim

        def build_one():
            layers = []
            last_dim = in_dim
            for h in hidden_dims:
                layers.append(nn.Linear(last_dim, h))
                if layer_norm:
                    layers.append(nn.LayerNorm(h))
                layers.append(activation())
                last_dim = h
            body = nn.Sequential(*layers) if layers else nn.Identity()
            head = nn.Linear(last_dim, 1)
            return body, head, last_dim

        body1, head1, _ = build_one()
        body2, head2, _ = build_one()

        self.q1_body = body1
        self.q1_head = head1
        self.q2_body = body2
        self.q2_head = head2

    def forward(self, obs: Tensor, goal: Tensor, act: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat([obs, goal, act], dim=-1)
        f1 = self.q1_body(x)
        f2 = self.q2_body(x)
        return self.q1_head(f1), self.q2_head(f2)


class GoalConditionedActor(nn.Module):
    """Gaussian policy pi(a | s, g)."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int],
        layer_norm: bool,
        activation: nn.Module,
        const_std: bool,
        min_log_std: float,
        max_log_std: float,
        state_dependent_std: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.const_std = const_std
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.state_dependent_std = not const_std if state_dependent_std is None else state_dependent_std

        hidden_dims = tuple(hidden_dims)
        in_dim = obs_dim + goal_dim

        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            last_dim = h

        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean = nn.Linear(last_dim, action_dim)

        if self.state_dependent_std:
            self.log_std_head = nn.Linear(last_dim, action_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: Tensor, goal: Tensor, temperature: float = 1.0) -> Tuple[Independent, Tensor, Tensor]:
        x = torch.cat([obs, goal], dim=-1)
        features = self.body(x)
        mean = self.mean(features)

        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std) * temperature
        dist = Independent(Normal(mean, std), 1)
        return dist, mean, std


def _to_tensor(value: Union[Tensor, np.ndarray, float, int], device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value.to(device)
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).to(device)
    elif isinstance(value, (float, int)):
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
    else:
        raise TypeError(f"Unsupported type {type(value)} for tensor conversion")

    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = _batch_to_device(value, device)
        else:
            result[key] = _to_tensor(value, device)
    return result


def _ensure_2d(tensor: Tensor) -> Tensor:
    # 支持 (B,) / (B,1) / (B,D) 等简单情况
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    return tensor


@dataclass
class GCIQLConfig:
    # learning rates
    lr: float = 3e-4
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    value_lr: Optional[float] = None

    # IQL hyper-params
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.9
    alpha: float = 0.3
    actor_loss: str = "ddpgbc"  # ["awr", "ddpgbc"]

    # nets
    actor_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    value_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    layer_norm: bool = True

    # policy std
    const_std: bool = True
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    state_dependent_std: Optional[bool] = None

    # others
    discrete: bool = False
    value_p_curgoal:  float = 0.2  # Probability of using the current state as the value goal.
    value_p_trajgoal: float = 0.5  # Probability of using a future state in the same trajectory as the value goal.
    value_p_randomgoal: float = 0.3  # Probability of using a random state as the value goal.
    value_geom_sample: bool = True  # Whether to use geometric sampling for future value goals.
    actor_p_curgoal: float = 0.0  # Probability of using the current state as the actor goal.
    actor_p_trajgoal: float = 1.0  # Probability of using a future state in the same trajectory as the actor goal.
    actor_p_randomgoal: float = 0.0  # Probability of using a random state as the actor goal.
    actor_geom_sample: bool = True  # Whether to use geometric sampling for future actor goals.
    gc_negative: bool = True  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
    p_aug: float = 0.0  # Probability of applying image augmentation.
    frame_stack: int = 0  # Number of frames to stack.


class GCIQLAgent:
    """PyTorch goal-conditioned IQL agent.

    使用方式刻意向 `MlpInvDynamic` 靠拢：
      - `update(batch)`：接收一个包含 IQL 训练所需字段的字典，返回若干 loss / 监控值
      - `predict(obs, goal)`：给定当前状态和目标，输出动作；可作为低层控制器使用
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        config: Optional[Union[Dict[str, Any], GCIQLConfig]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim or obs_dim

        if config is None:
            cfg = GCIQLConfig()
        elif isinstance(config, GCIQLConfig):
            cfg = config
        else:
            # Dict 或 DictConfig -> 先转成普通 dict，再解包到 dataclass
            if isinstance(config, DictConfig):
                cfg_dict = dict(config)
            else:
                cfg_dict = dict(config)
            cfg = GCIQLConfig(**cfg_dict)
        self.config = cfg

        if self.config.discrete:
            raise NotImplementedError("Discrete action spaces are not supported in this implementation.")

        activation_cls = nn.Mish

        # V(s, g)
        self.value_net = GoalConditionedValue(
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            hidden_dims=self.config.value_hidden_dims,
            layer_norm=self.config.layer_norm,
            activation=activation_cls,
        ).to(self.device)

        # Q(s, a, g)
        self.critic = GoalConditionedCritic(
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.value_hidden_dims,
            layer_norm=self.config.layer_norm,
            activation=activation_cls,
        ).to(self.device)

        # target Q
        self.target_critic = GoalConditionedCritic(
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.value_hidden_dims,
            layer_norm=self.config.layer_norm,
            activation=activation_cls,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # pi(a | s, g)
        self.actor = GoalConditionedActor(
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.actor_hidden_dims,
            layer_norm=self.config.layer_norm,
            activation=activation_cls,
            const_std=self.config.const_std,
            min_log_std=self.config.min_log_std,
            max_log_std=self.config.max_log_std,
            state_dependent_std=self.config.state_dependent_std,
        ).to(self.device)

        lr = self.config.lr
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.config.value_lr or lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr or lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr or lr)

    # ----------------- public API (类似 invdynamic) -----------------
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """更新 GCIQL 参数。假设 batch 已包含训练 IQL 所需的字段。"""
        batch_torch = _batch_to_device(batch, self.device)

        value_metrics = self._update_value(batch_torch)
        critic_metrics = self._update_critic(batch_torch)
        actor_metrics = self._update_actor(batch_torch)

        # EMA 更新 target Q
        self._soft_update(self.critic, self.target_critic, self.config.tau)

        info: Dict[str, float] = {}
        info.update({f"value/{k}": v for k, v in value_metrics.items()})
        info.update({f"critic/{k}": v for k, v in critic_metrics.items()})
        info.update({f"actor/{k}": v for k, v in actor_metrics.items()})
        return info

    @torch.no_grad()
    def predict(
        self,
        obs: Union[Tensor, np.ndarray],
        goal: Union[Tensor, np.ndarray],
        deterministic: bool = True,
    ) -> Tensor:
        """给定 (obs, goal) 输出动作，接口类似 MlpInvDynamic.predict。"""
        obs_tensor = _to_tensor(obs, self.device)
        goal_tensor = _to_tensor(goal, self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if goal_tensor.ndim == 1:
            goal_tensor = goal_tensor.unsqueeze(0)

        dist, mean, _ = self.actor(obs_tensor, goal_tensor, temperature=1.0)
        act = mean if deterministic else dist.sample()
        return torch.clamp(act, -1.0, 1.0)

    def __call__(self, obs: Union[Tensor, np.ndarray], goal: Union[Tensor, np.ndarray]) -> Tensor:
        return self.predict(obs, goal)

    def train(self) -> None:
        self.actor.train()
        self.critic.train()
        self.value_net.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.value_net.eval()

    def save(self, path: str) -> None:
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "value": self.value_net.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "value_opt": self.value_optimizer.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(payload, path)

    def load(self, path: str, load_optimizers: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        self.value_net.load_state_dict(payload["value"])

        if load_optimizers:
            if "actor_opt" in payload:
                self.actor_optimizer.load_state_dict(payload["actor_opt"])
            if "critic_opt" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_opt"])
            if "value_opt" in payload:
                self.value_optimizer.load_state_dict(payload["value_opt"])

    # ----------------- 内部训练步骤 -----------------
    def _update_value(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        actions = batch["actions"]
        goals = self._resolve_goal(batch, "value_goals", ("actor_goals", "goals"))

        with torch.no_grad():
            q1, q2 = self.target_critic(obs, goals, actions)
            q = torch.min(q1, q2)

        v = self.value_net(obs, goals)
        diff = q - v
        loss = self._expectile_loss(diff, self.config.expectile)

        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.value_optimizer.step()

        return {
            "value_loss": float(loss.item()),
            "v_mean": float(v.mean().item()),
            "v_max": float(v.max().item()),
            "v_min": float(v.min().item()),
        }

    def _update_critic(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        rewards = _ensure_2d(batch["rewards"])
        masks = _ensure_2d(self._resolve_mask(batch))
        goals = self._resolve_goal(batch, "value_goals", ("actor_goals", "goals"))

        with torch.no_grad():
            next_v = self.value_net(next_obs, goals)
            target_q = rewards + self.config.discount * masks * next_v

        q1, q2 = self.critic(obs, goals, actions)
        loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()

        q = torch.min(q1, q2)
        return {
            "critic_loss": float(loss.item()),
            "q_mean": float(q.mean().item()),
            "q_max": float(q.max().item()),
            "q_min": float(q.min().item()),
        }

    def _update_actor(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        actions = batch["actions"]
        goals = self._resolve_goal(batch, "actor_goals", ("value_goals", "goals"))

        dist, mean, std = self.actor(obs, goals, temperature=1.0)
        log_prob = dist.log_prob(actions)

        if self.config.actor_loss == "awr":
            with torch.no_grad():
                v = self.value_net(obs, goals)
                q1, q2 = self.critic(obs, goals, actions)
                q = torch.min(q1, q2)
                adv = q - v
            weights = torch.exp(torch.clamp(adv * self.config.alpha, max=10.0))
            loss = -(weights * log_prob).mean()
            metrics = {
                "actor_loss": float(loss.item()),
                "adv": float(adv.mean().item()),
                "bc_log_prob": float(log_prob.mean().item()),
                "mse": float(((mean - actions) ** 2).mean().item()),
                "std": float(std.mean().item()),
            }
        elif self.config.actor_loss == "ddpgbc":
            q_actions = mean if self.config.const_std else dist.rsample()
            q_actions = torch.clamp(q_actions, -1.0, 1.0)
            q1, q2 = self.critic(obs, goals, q_actions)
            q = torch.min(q1, q2).detach()

            q_scale = q.abs().mean().clamp(min=1e-6)
            q_loss = -q.mean() / q_scale
            bc_loss = -(self.config.alpha * log_prob).mean()
            loss = q_loss + bc_loss
            metrics = {
                "actor_loss": float(loss.item()),
                "q_loss": float(q_loss.item()),
                "bc_loss": float(bc_loss.item()),
                "q_mean": float(q.mean().item()),
                "q_abs_mean": float(q.abs().mean().item()),
                "bc_log_prob": float(log_prob.mean().item()),
                "mse": float(((mean - actions) ** 2).mean().item()),
                "std": float(std.mean().item()),
            }
        else:
            raise ValueError(f"Unsupported actor_loss: {self.config.actor_loss}")

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        return metrics

    # ----------------- small helpers -----------------
    @staticmethod
    def _expectile_loss(diff: Tensor, expectile: float) -> Tensor:
        weight = torch.where(diff >= 0, expectile, 1 - expectile)
        return (weight * diff.pow(2)).mean()

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for src, tgt in zip(source.parameters(), target.parameters()):
                tgt.data.mul_(1.0 - tau).add_(tau * src.data)

    def _resolve_goal(self, batch: Dict[str, Tensor], primary: str, fallbacks: Tuple[str, ...]) -> Tensor:
        if primary in batch:
            return batch[primary]
        for key in fallbacks:
            if key in batch:
                return batch[key]
        raise KeyError(f"Goal key '{primary}' not found in batch and no fallback available.")

    @staticmethod
    def _resolve_mask(batch: Dict[str, Tensor]) -> Tensor:
        if "masks" in batch:
            return batch["masks"]
        if "dones" in batch:
            return 1.0 - batch["dones"]
        return torch.ones_like(batch["rewards"])


def get_config() -> Dict[str, Any]:
    """默认超参字典，可给 Hydra 用作参考。"""
    return asdict(GCIQLConfig())