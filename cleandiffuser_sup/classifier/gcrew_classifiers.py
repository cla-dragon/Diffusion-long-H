import torch
from typing import Optional

from cleandiffuser.nn_classifier import BaseNNClassifier
from cleandiffuser.classifier.base import BaseClassifier

# Classifier that calculate the goal distance in goal-conditioned settings: Q(s, a, g)
# Adapted from CumRewClassifier
class GCDistance(BaseClassifier):
    def __init__(
            self,
            nn_classifier: BaseNNClassifier,
            device: str = "cpu",
            optim_params: Optional[dict] = None,
            distance_dims: Optional[torch.Tensor] = None,
    ):
        super().__init__(nn_classifier, 0.995, None, optim_params, device)
        self.distance_dims = distance_dims

    def loss(self, x, noise, R):
        return 0

    def update(self, x, noise, R):
        self.optim.zero_grad()
        loss = self.loss(x, noise, R)
        loss.backward()
        self.optim.step()
        self.ema_update()
        return {"loss": loss.item()}

    def logp(self, x, noise, c=None):
        if self.distance_dims is not None:
            diff = (x - c)[:, :, self.distance_dims] # (B, T, state_dim)
        else:
            diff = x - c
        dist = (diff ** 2).mean(-1, keepdim=True)
        return -dist