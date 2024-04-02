from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["BinaryCrossEntropyLoss"]


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, output_key: str, target_key: str, rescale: Optional[float] = None) -> None:
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.rescale = rescale

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        output = data[self.output_key]
        target = data[self.target_key]

        if self.rescale is not None:
            weight = torch.where(torch.isclose(target, torch.ones_like(target)), self.rescale, 1)
        else:
            weight = None

        return F.binary_cross_entropy_with_logits(output, target, weight=weight)
