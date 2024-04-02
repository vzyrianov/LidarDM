from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["SoftmaxPlanLoss"]


class SoftmaxPlanLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        output = data["predicted_plan"]
        output = output.reshape((output.shape[0] * output.shape[1], -1))

        target = data["plan"]
        target = target.reshape((target.shape[0] * target.shape[1], -1))
        
        return F.cross_entropy(output, target)
