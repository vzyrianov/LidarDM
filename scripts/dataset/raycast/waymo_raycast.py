import os
import gzip
import argparse
from natsort import natsorted
from tqdm import tqdm

import numpy as np
import torch

from lidardm.visualization.meshify import FieldsMeshifier
from lidardm.lidar_generation.scene_composition.compositors import WaymoCompositor
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Script dump raycasted results given tsdf fields from Waymo')
  parser.add_argument('--split', type=str, nargs='+', help='split')
  parser.add_argument('--fields_path', type=str, help='path to fieldds')
  parser.add_argument('--dataset_path', type=str, help='path to preprocessed dataset')
  parser.add_argument('--out_dir', type=str, help='output folder')

  parser.add_argument('--start_scan', type=int, default=0)
  parser.add_argument('--end_scan', type=int, default=100000)
  args = parser.parse_args()
 
  device = torch.device("cuda:0")
  meshifier = FieldsMeshifier(device, 0.15, [96, 96, 6], use_post_process=False)

  for split_type in args.split:
    field_master_folder = args.fields_path
    field_master_folder = os.path.join(field_master_folder, split_type)
    tfrecords = natsorted(os.listdir(field_master_folder)) 
  
    for tfrecord_idx in tqdm(range(args.start_scan, min(args.end_scan+1, len(tfrecords)))):
      tfrecord = tfrecords[tfrecord_idx]
      print("Processing", tfrecord)

      field_folder = os.path.join(field_master_folder, tfrecord, 'grid')
      field_files = natsorted(os.listdir(field_folder)) 

      output_folder = os.path.join(args.out_dir, split_type, tfrecord, 'raycast')

      # skip if exists
      if os.path.isdir(output_folder):
        continue

      os.makedirs(output_folder)
      print("Saving raycast to", output_folder)
      
      for i in tqdm(range(len(field_files))):
        start_end = field_files[i][:-7]
        start_frame, end_frame = [int(s) for s in start_end.split('-') if s.isdigit()]
        center_frame = (end_frame - start_frame) // 2 + start_frame

        input_file = os.path.join(field_folder, field_files[i])
        f = gzip.GzipFile(input_file, "r")
        field = np.load(f)
        f.close()

        field = torch.from_numpy(field).cuda()
        mesh = meshifier.generate_mesh(field)
        
        compositor = WaymoCompositor(mesh=mesh,
                                     sequence_id=tfrecord, 
                                     anchor_frame_idx=center_frame,
                                     n_frames=1,
                                     waymo_dataset_root=args.dataset_path)

        scene = compositor.__getitem__(0)
  
        raycast = scene["scan"]
        np.save(os.path.join(output_folder, f'{center_frame}.npy'), raycast)
  
      
        
  
        

        