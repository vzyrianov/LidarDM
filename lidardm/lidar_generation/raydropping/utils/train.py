#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil

from .trainer import Trainer
from lidardm import PROJECT_DIR

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./train.py")
  # parser.add_argument('--raycast_path', type=str, required=True, help='Path to raycast')
  # parser.add_argument('--raw_path', type=str, required=True, help='Path to raw')
  parser.add_argument('--arch_cfg', type=str, required=True, help='architecture yaml file')
  parser.add_argument('--log', type=str,
      default=os.path.expanduser("~") + '/logs/' +
      datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
      help='Directory to put the log data. Default: ~/logs/date+time'
  )
  parser.add_argument('--pretrained', type=str, required=False, default=None, help='pretrained path')

  FLAGS, unparsed = parser.parse_known_args()

  if 'waymo' in  FLAGS.arch_cfg:
    dataset = 'waymo'
    raycast_path = os.path.join(PROJECT_DIR, '_datasets', 'waymo_raycast')
    raw_path = os.path.join(PROJECT_DIR, '_datasets', 'waymo_preprocessed')
  elif 'kitti360' in FLAGS.arch_cfg:
    dataset = 'kitti360'
    raycast_path = os.path.join(PROJECT_DIR, '_datasets', 'kitti_raycast')
    raw_path = os.path.join(PROJECT_DIR, '_datasets', 'kitti')
  else:
    raise ValueError('not supported')
  
  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset", dataset)
  print("raycast_path", raycast_path)
  print("raw_path", raw_path)
  print("arch_cfg", FLAGS.arch_cfg)
  print("log", FLAGS.log)
  print("pretrained", FLAGS.pretrained)
  print("----------\n")
  print("Commit hash (training version): ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

  # open arch config file
  try:
    print("Opening arch config file %s" % FLAGS.arch_cfg)
    ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # create log folder
  try:
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  # does model folder exist?
  if FLAGS.pretrained is not None:
    if os.path.isdir(FLAGS.pretrained):
      print("model folder exists! Using model from %s" % (FLAGS.pretrained))
    else:
      print("model folder doesnt exist! Start with random weights...")
  else:
    print("No pretrained directory found.")

  # copy all files to log folder (to remember what we did, and make inference
  # easier). Also, standardize name to be able to open it later
  try:
    print("Copying files to %s for further reference." % FLAGS.log)
    copyfile(FLAGS.arch_cfg, FLAGS.log + "/arch_cfg.yaml")
  except Exception as e:
    print(e)
    print("Error copying files, check permissions. Exiting...")
    quit()

  # create trainer and start the training
  trainer = Trainer(ARCH, dataset, raycast_path, raw_path, FLAGS.log, FLAGS.pretrained)
  trainer.train()
