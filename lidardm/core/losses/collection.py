from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

__all__ = ["LossCollection"]


class LossCollection(nn.ModuleDict):
    def __init__(
        self,
        modules: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(modules)
        self.weights = weights or {}

    def forward(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {key: val(data) for key, val in self.items()}
        loss = sum(self.weights.get(key, 1) * val for key, val in losses.items())
        return loss, losses
