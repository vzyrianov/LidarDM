import os
import gzip
import argparse
from natsort import natsorted
from tqdm import tqdm

import numpy as np
import torch
import open3d as o3d

from kitti360scripts.helpers.annotation import Annotation3D

from lidardm.visualization.meshify import FieldsMeshifier
from lidardm.lidar_generation.scene_composition.compositors.scene_compositor import SceneCompositor, AgentDict

trans_velo_to_imu = np.array(
        [[0.99992906, 0.0057743, 0.01041756, 0.77104934],
         [0.00580536, -0.99997879, -0.00295331, 0.29854144],
         [0.01040029, 0.00301357, -0.99994137, -0.83628022],
         [0, 0, 0, 1]])

vehicle_type = [26, 27, 28, 29, 30, 43]
pedestrian_type = [24, 25]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Script dump raycasted results given tsdf fields from KITTI360')
  parser.add_argument('--seq', type=int, nargs='+', help='sequence', default=0)
  parser.add_argument('--fields_path', type=str, help='path to fields')
  parser.add_argument('--dataset_path', type=str, help='path to fields')
  parser.add_argument('--out_dir', type=str, help='output folder')
  
  parser.add_argument('--start_scan', type=int, default=0)
  parser.add_argument('--end_scan', type=int, default=100000)
  args = parser.parse_args()

  device = torch.device("cuda:0")
  meshifier = FieldsMeshifier(device, 0.15, [96, 96, 6], use_post_process=False)
  os.environ["KITTI360_DATASET"] = args.dataset_path

  for seq in args.seq:
    print("Processing Sequence", seq)
   
    sequence_name = '2013_05_28_drive_%04d_sync' % seq

    field_folder = os.path.join(args.fields_path, sequence_name, 'grid')
    field_files = natsorted(os.listdir(args.fields_path)) 

    bbox_folder = os.path.join(args.dataset_path, 'data_3d_bboxes')
    annotation3D = Annotation3D(bbox_folder, sequence_name)
    
    pose_file = os.path.join(args.dataset_path, 'data_interp_poses', sequence_name, 'poses.txt')
    poses = np.loadtxt(pose_file, delimiter=' ', dtype=float)
    pose_frames, poses = poses[:,0], poses[:,1:]

    output_dir = "" if args.out_dir is None else args.out_dir
    output_folder = os.path.join(output_dir, sequence_name)

    # skip if exists
    if os.path.isdir(output_folder):
      continue

    os.makedirs(output_folder)
    print("Saving raycast to", output_folder)

    for i in tqdm(range(args.start_scan, min(args.end_scan+1, len(field_files)))): 
      start_end = field_files[i][:-7]
      start_frame, end_frame = [int(s) for s in start_end.split('-') if s.isdigit()]
      center_frame = (end_frame - start_frame) // 2 + start_frame

      input_file = os.path.join(field_folder, field_files[i])
      f = gzip.GzipFile(input_file, "r")
      field = np.load(f)
      f.close()

      field = torch.from_numpy(field).cuda()
      mesh = meshifier.generate_mesh(field)

      # velo -> world
      pose = poses[center_frame,:].reshape(4,4) @ trans_velo_to_imu

      # get all agents at the current center frame
      agent_dict = AgentDict
      for globalId,v in annotation3D.objects.items():
        for obj in v.values():
          if obj.timestamp != -1 and obj.timestamp == center_frame:
            # create a 3D bbox mesh
            bbox_corners = o3d.geometry.PointCloud()
            bbox_corners.points = o3d.utility.Vector3dVector(obj.vertices)
            bbox_corners.transform(np.linalg.inv(pose)) # back to ego frame

            if obj.semanticId in vehicle_type:
              label = 'Vehicle'
            elif obj.semanticId in pedestrian_type:
              label = 'Pedestrian'
            else:
              continue
            
            agent_dict.insert_agent(agent_id=globalId, 
                                    semantic=label, 
                                    bboxes={0: (np.asarray(bbox_corners.points), None)})

      scene_compositor = SceneCompositor(raycaster_config='kitti360', mesh=mesh)
      scene_compositor.update_scenarios(poses=[np.eye(4)], agents=agent_dict)
      scene = scene_compositor.__getitem__(0)

      raycast = scene["scan"]
      np.save(os.path.join(output_folder, f'{center_frame}.npy'), raycast)
  