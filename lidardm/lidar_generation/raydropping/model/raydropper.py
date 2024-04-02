#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import lidardm.lidar_generation.raydropping.model.backbones as backbones
import lidardm.lidar_generation.raydropping.model.decoders as decoders

class RayDropper(nn.Module):
  def __init__(self, arch_config, pretrained_path=None, pretrained_suffix=""):
    super().__init__()
    self.arch_config = arch_config
    self.strict = False

    # backbone module
    if self.arch_config["backbone"]["name"] == 'darknet':
      self.backbone = backbones.DarkNet(params=self.arch_config["backbone"])
    elif self.arch_config["backbone"]["name"] == 'squeezeseg':
      self.backbone = backbones.SqueezeSeg(params=self.arch_config["backbone"])
    elif self.arch_config["backbone"]["name"] == 'squeezesegV2':
      self.backbone = backbones.SqueezeSegV2(params=self.arch_config["backbone"])

    # do a pass of the backbone to initialize the skip connections
    stub = torch.zeros((1,
                        self.backbone.get_input_depth(),
                        self.arch_config["dataset"]["img_prop"]["height"],
                        self.arch_config["dataset"]["img_prop"]["width"]))

    if torch.cuda.is_available():
      stub = stub.cuda()
      self.backbone.cuda()
    _, stub_skips = self.backbone(stub)

    decoder_args = {
      "params": self.arch_config["decoder"],
      "stub_skips": stub_skips,
      "OS": self.arch_config["backbone"]["OS"],
      "feature_depth": self.backbone.get_last_depth()
    }

    # decoder module
    if self.arch_config["backbone"]["name"] == 'darknet':
      self.decoder = decoders.DarkNet(**decoder_args)
    elif self.arch_config["backbone"]["name"] == 'squeezeseg':
      self.decoder = decoders.SqueezeSeg(**decoder_args)
    elif self.arch_config["backbone"]["name"] == 'squeezesegV2':
      self.decoder = decoders.SqueezeSegV2(**decoder_args)

    # head module 
    self.head = nn.Sequential(nn.Dropout2d(p=arch_config["head"]["dropout"]),
                              nn.Conv2d(self.decoder.get_last_depth(),
                                        out_channels=1, kernel_size=3,
                                        stride=1, padding=1))
    # train backbone?
    if not self.arch_config["backbone"]["train"]:
      for w in self.backbone.parameters():
        w.requires_grad = False

    # train decoder?
    if not self.arch_config["decoder"]["train"]:
      for w in self.decoder.parameters():
        w.requires_grad = False

    # train head?
    if not self.arch_config["head"]["train"]:
      for w in self.head.parameters():
        w.requires_grad = False

    # get weights
    if pretrained_path is not None:
      # try backbone
      try:
        w_dict = torch.load(pretrained_path + "/backbone" + pretrained_suffix,
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print("Couldn't load backbone module. Check pathname")
        raise e

      # try decoder
      try:
        w_dict = torch.load(pretrained_path + "/decoder" + pretrained_suffix,
                            map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model decoder weights")
      except Exception as e:
        print("Couldn't load decoder module. Check pathname")
        raise e

      # try head
      try:
        w_dict = torch.load(pretrained_path + "/head" + pretrained_suffix,
                            map_location=lambda storage, loc: storage)
        self.head.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head module. Check pathname")
        raise e

    else:
      print("No path to pretrained, using random init.")

  def forward(self, x, mask=None):
    y, skips = self.backbone(x)
    y = self.decoder(y, skips)
    y = self.head(y)
    return y

  def save_checkpoint(self, logdir, suffix=""):
    # Save the weights
    torch.save(self.backbone.state_dict(), logdir +
               "/backbone" + suffix)
    torch.save(self.decoder.state_dict(), logdir +
               "/decoder" + suffix)
    torch.save(self.head.state_dict(), logdir +
               "/head" + suffix)
