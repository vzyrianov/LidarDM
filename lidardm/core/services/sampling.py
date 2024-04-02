import os
import numpy as np

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt

from lidardm.core.models.diffusion import ScheduleConfig

from lidardm.visualization import unvoxelize_field
from lidardm.core.datasets.utils import voxelize

from lidardm.core.schedulers.modified_schedulers import *
from lidardm.core.datasets.utils import unscale_field
from lidardm.visualization.meshify import FieldsMeshifier
from hydra.utils import instantiate

def instantiate_conditional_model(vae_path, map_vae_path, unet_path):
    cfg = {
        'model': {
            '_target_': 'lidardm.core.models.DiffusionPipelineFieldCond',
            'autoencoder': {
                '_target_': 'lidardm.core.models.VAEFields',
                'num_channels': 40,
                'latent_channels': 8,
                'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
                'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
                'layers_per_block': 2, 'block_out_channels': [128, 128, 256, 512, 512],
                'pretrained': vae_path
            },
            'unet': {
                '_target_': 'lidardm.core.models.unet.UNet2DModelSimpleConditional',
                'sample_size': 64,
                'in_channels': 12,
                'out_channels': 8,
                'down_block_types': ['DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'],
                'up_block_types': ['AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'],
                'block_out_channels': [448, 640, 896, 1280],
                'layers_per_block': 2,
                'attention_head_dim': 8,
                'norm_num_groups': 32,
                'pretrained': unet_path
            }, 
            'map_autoencoder': {
                '_target_': 'lidardm.core.models.MapVAE',
                'num_channels': 9,
                'layers_per_block': 1,
                'block_out_channels': [64, 64, 128, 256, 512],
                'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
                'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
                'pretrained': map_vae_path
            }, 
            'channels': 8
        }
    }

    model =  instantiate(DictConfig(cfg))

    return model.model



def sample_from_map(map_cond, cond_model, sampling_steps=600):
    CFG_SCALE = 4.0
    scheduler = ModifiedEulerAScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')

    # Provides batch size, map, and device information
    data_map = {
        'lidar': torch.zeros((1, 1, 1, 1)).cuda(),
        'field': torch.zeros((1, 1, 1, 1)).cuda(),
        'bev': torch.unsqueeze(map_cond, 0).cuda()
    }

    schedule_config = ScheduleConfig()
    schedule_config.last_noise_samples = 10
    schedule_config.second_to_last_noise_samples = 8
    schedule_config.use_noise_second_to_last = True
    schedule_config.use_noise_last = False

    generated_map = cond_model.forward_test(data_map, scheduler=scheduler, gumbel_tau=1.0, schedule_config=schedule_config, cfg_scale=CFG_SCALE,
                                    use_gumbel=False, threshold=0.3, num_steps=sampling_steps, generate_video=False)
    
    field_recon = unscale_field(generated_map['video_field'][0], -1.0, 1.0)
            
    unvoxelized = unvoxelize_field(torch.ones_like(field_recon), field_recon, [-47.999, 48.001, -47.999, 48.001, -2.999, 3.001], [0.15, 0.15, 0.15])

    device = torch.device("cuda:0")
    meshifier = FieldsMeshifier(device, 0.15, [96, 96, 6])
    mesh = meshifier.generate_mesh(unvoxelized.to(device))

    return mesh
    


def instantiate_unconditional_model_kitti(vae_path, unet_path):
    cfg = {
        'model': {
            '_target_': 'lidardm.core.models.DiffusionPipelineField',
            'autoencoder': {
                '_target_': 'lidardm.core.models.VAEFields',
                'num_channels': 40,
                'latent_channels': 8,
                'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
                'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
                'layers_per_block': 2, 'block_out_channels': [128, 128, 256, 512, 512],
                'pretrained': vae_path  
            },
            'unet': {
                '_target_': 'lidardm.core.models.unet.UNet2DModelSimple',
                'sample_size': 64,
                'in_channels': 8,
                'out_channels': 8,
                'down_block_types': ['DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'],
                'up_block_types': ['AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'],
                'block_out_channels': [448, 640, 896, 1280],
                'layers_per_block': 2,
                'attention_head_dim': 8,
                'norm_num_groups': 32,
                'pretrained': unet_path
            },
            'channels': 8
        }
    }

    model =  instantiate(DictConfig(cfg))

    return model.model

def sample_uncond_kitti(model, sampling_steps=600):
    scheduler = ModifiedEulerAScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')
    #scheduler = ModifiedDPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')

    # Provides batch size, map, and device information
    data_map = {
        'lidar': torch.zeros((1, 1, 1, 1)).cuda(),
        'field': torch.zeros((1, 1, 1, 1)).cuda()
    }

    schedule_config = ScheduleConfig()
    schedule_config.last_noise_samples = 10
    schedule_config.second_to_last_noise_samples = 8
    schedule_config.use_noise_second_to_last = True
    schedule_config.use_noise_last = False

    generated_map = model.forward_test(data_map, scheduler=scheduler, gumbel_tau=1.0, schedule_config=schedule_config,
                                    use_gumbel=False, threshold=0.3, num_steps=sampling_steps, generate_video=False)
    
    field_recon = unscale_field(generated_map['video_field'][0], -1.0, 1.0)
            
    unvoxelized = unvoxelize_field(torch.ones_like(field_recon), field_recon, [-47.999, 48.001, -47.999, 48.001, -2.999, 3.001], [0.15, 0.15, 0.15])

    device = torch.device("cuda:0")
    meshifier = FieldsMeshifier(device, 0.15, [96, 96, 6])
    mesh = meshifier.generate_mesh(unvoxelized.to(device))

    return mesh