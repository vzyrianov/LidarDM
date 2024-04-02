
from diffusers import UNet2DModel
from .utils import load_checkpoint
from typing import Any, Dict, Tuple
import torch.nn as nn
import torch
from diffusers.models.unet_2d_blocks import ResnetDownsampleBlock2D, Upsample2D, DownEncoderBlock2D

__all__ = ["PlanningResnet"]


class NeuralMotionPlannerArchitecture(nn.Module):
    def __init__(self, in_channels, out_channels, down_channels, up_channels, groups=8) -> None:
        super().__init__()
        
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=(1, 1))
        self.conv_norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=groups)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, padding=1)

        for i in range(0, len(down_channels)):
            down_block = DownEncoderBlock2D(in_channels=down_channels[i],
                                      out_channels=down_channels[i+1] if i+1<len(up_channels) else up_channels[0],
                                      resnet_groups=groups)
            self.down_blocks.append(down_block)
        

        for i in range(0, len(up_channels)):
            up_block = Upsample2D(channels=down_channels[-1] if i == 0 else up_channels[i-1],
                                  out_channels=up_channels[i],
                                  use_conv=True)
            
            self.up_blocks.append(up_block)
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        sample = x

        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        for up_block in self.up_blocks: 
            sample = up_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class PlanningResnet(NeuralMotionPlannerArchitecture):
    def __init__(self, pretrained=None, append_map=False, **kwargs) -> None:
        super().__init__(**kwargs)

        self.append_map = append_map

        if pretrained is not None:
            load_checkpoint(self, pretrained)

    def forward(self, x: Dict[str, any]):
        inputs = torch.cat([x["lidar_seq"][:,i] for i in range(x["lidar_seq"].shape[1])], dim=1)

        if(self.append_map):
            inputs = torch.cat([inputs, x["bev"]], dim=1)

        output = super().forward(inputs)
        x["predicted_plan"] = output
        return x

