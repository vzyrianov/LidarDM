from diffusers import UNet2DModel
import torch
from .utils import load_checkpoint

__all__ = ["UNet2DModelSimple", "Net2DModelSimpleConditional"]

class UNet2DModelSimple(UNet2DModel):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        if pretrained is not None:
            load_checkpoint(self, pretrained)
    
    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
        return super().forward(x, t)
    
class UNet2DModelSimpleConditional(UNet2DModelSimple):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, t) -> torch.Tensor:
        x = torch.cat((x,cond), dim=1)
        return super().forward(x, t)
