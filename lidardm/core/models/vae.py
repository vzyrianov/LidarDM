from typing import Any, Dict, Optional, Tuple

import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder

from .utils import load_checkpoint
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

__all__ = ["LiDARVAE", "MapVAE", "VAEFields"]


class VAE(AutoencoderKL):
    def __init__(
        self,
        num_channels: int = 35,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",) * 4,
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",) * 4,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 512,
        scaling_factor: float = 0.18215,
        pretrained: Optional[str] = None,
    ) -> None:
        super().__init__(
            in_channels=num_channels,
            out_channels=num_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
        )
        if pretrained is not None:
            load_checkpoint(self, pretrained)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        latent = super().encode(x).latent_dist
        recon = super().decode(latent.sample()).sample
        return {"latent": latent, "recon": recon}


class LiDARVAE(VAE):
    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        outputs = super().forward(data["lidar"])
        for key, value in outputs.items():
            data["lidar/" + key] = value
        return data
    
class MapVAE(VAE):
    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        outputs = super().forward(data["bev"])
        for key, value in outputs.items():
            data["bev/" + key] = value
        return data
    
        
    
class VAEFields(VAE):
    def __init__(
        self,
        num_channels: int = 40,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",) * 4,
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",) * 4,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 512,
        scaling_factor: float = 0.18215,
        pretrained: Optional[str] = None,
    ) -> None:
        self.num_channels = num_channels
        super().__init__(
            num_channels=num_channels*1,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            pretrained=pretrained
        )

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        outputs = super().forward(data['field'])

        #data["lidar/recon"] = outputs['recon'][:,:self.num_channels]
        data["field/recon"] = outputs['recon'][:,:]
        data["lidar/latent"] = outputs['latent']

        return data


