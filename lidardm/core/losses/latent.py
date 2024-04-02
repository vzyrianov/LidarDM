from typing import Any, Dict

import torch
from torch import nn

__all__ = ["KLDivergenceLoss"]


class KLDivergenceLoss(nn.Module):
    def __init__(self, latent_key: str) -> None:
        super().__init__()
        self.latent_key = latent_key

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        latent = data[self.latent_key]
        return latent.kl().mean()
