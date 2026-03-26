from typing import Dict, Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_condition import BaseNNCondition, get_mask
from cleandiffuser.utils import at_least_ndim


class SegmentBoundaryCondition(BaseNNCondition):
    """Encode left/right boundary payloads into a single condition vector."""

    def __init__(
        self,
        obs_dim: int,
        overlap_steps: int,
        emb_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.overlap_steps = int(overlap_steps)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim or emb_dim * 2)
        self.dropout = float(dropout)

        side_input_dim = self.overlap_steps * self.obs_dim + 4
        self.left_encoder = self._make_side_encoder(side_input_dim)
        self.right_encoder = self._make_side_encoder(side_input_dim)
        self.fuse = nn.Sequential(
            nn.Linear(self.emb_dim * 2 + 6, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.emb_dim),
        )

    def _make_side_encoder(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.emb_dim),
        )

    def _encode_side(
        self,
        encoder: nn.Module,
        chunk: torch.Tensor,
        level: torch.Tensor,
        state_code: torch.Tensor,
    ) -> torch.Tensor:
        flat_chunk = chunk.reshape(chunk.shape[0], -1)
        level = level.reshape(chunk.shape[0], 1)
        features = torch.cat([flat_chunk, level, state_code], dim=-1)
        return encoder(features)

    def forward(self, condition: Dict[str, torch.Tensor], mask: torch.Tensor = None):
        if condition is None:
            raise ValueError("SegmentBoundaryCondition requires a condition payload.")

        left_chunk = condition["left_chunk"]
        right_chunk = condition["right_chunk"]
        left_level = condition["left_level"]
        right_level = condition["right_level"]
        left_state = condition["left_state"]
        right_state = condition["right_state"]

        left_emb = self._encode_side(self.left_encoder, left_chunk, left_level, left_state)
        right_emb = self._encode_side(self.right_encoder, right_chunk, right_level, right_state)
        fused = self.fuse(torch.cat([left_emb, right_emb, left_state, right_state], dim=-1))

        if self.dropout > 0.0:
            fused_mask = at_least_ndim(
                get_mask(mask, (fused.shape[0],), self.dropout, self.training, fused.device),
                fused.dim(),
            )
            fused = fused * fused_mask

        return fused

