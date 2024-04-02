from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["SigmoidMaskMSELoss", "SigmoidMSELoss"]

class SigmoidMaskMSELoss(nn.Module):
    def __init__(self, output_key: str, target_key: str) -> None:
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        output = data[self.output_key]
        target = data[self.target_key]
        return (((F.sigmoid(output) * data['lidar']) - (target * data['lidar']))**2).sum() / (data['lidar'].sum() + 1)

class SigmoidMSELoss(nn.Module):
    def __init__(self, output_key: str, target_key: str) -> None:
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        output = data[self.output_key]
        target = data[self.target_key]
        return ((F.sigmoid(output) - target)**2).sum() / output.shape[0]
