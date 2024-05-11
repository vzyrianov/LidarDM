#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import numpy as np
import yaml 
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from lidardm.lidar_generation.raydropping.model.raydropper import RayDropper
from lidardm.core.models.gumbel_sigmoid import gumbel_sigmoid
from lidardm import PROJECT_DIR

RAYDROPPING_DIR = Path(__file__).parents[1].absolute()

class RayDropInferer():
  def __init__(self, dataset, pretrained_suffix=".pt", threshold=0.5):

    supported_datasets = ['kitti360', 'waymo']

    if dataset not in supported_datasets:
      raise KeyError(f'{dataset} not supported')

    arch_config = yaml.safe_load(open(os.path.join(RAYDROPPING_DIR, 
                                                    'config',
                                                    f'{dataset}.yaml'), 'r'))
    
    pretrained_path = os.path.join(PROJECT_DIR, 'pretrained_models', dataset, 'raydrop')
    if not os.path.isdir(pretrained_path):
      raise KeyError(f"Pretrained model for {dataset} raydrop does not exist.")

    # parameters
    self.arch_config = arch_config
    self.pretrained_path = pretrained_path
    self.pretrained_suffix = pretrained_suffix
    self.dataset = dataset
    self.threshold = threshold

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = RayDropper(self.arch_config, self.pretrained_path, self.pretrained_suffix)

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self, raycast_im, gumbel=True):

    raycast_im = torch.as_tensor(np.array(raycast_im)).to(dtype=torch.float)
    raycast_im = raycast_im.unsqueeze(0).unsqueeze(0).cuda()

    # validation mode
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      pred_mask = self.model(raycast_im)
      if gumbel:
        pred_mask = gumbel_sigmoid(pred_mask, 1, 0, tau=1, threshold=self.threshold, hard=True)
      else:
        pred_mask = nn.functional.sigmoid(pred_mask) > self.threshold

    return pred_mask.detach().clone().cpu().numpy()[0,0]
      
  
  
      