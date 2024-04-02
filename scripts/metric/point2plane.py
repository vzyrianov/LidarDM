from glob import glob
import os
import open3d as o3d
import torch
import numpy as np
from re import I
import time
import cv2
import numpy as np
import open3d as o3d
import argparse
import time
import sys
from sklearn.neighbors import KDTree

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

def get_normal(pc1):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pc1)
  pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  return np.asarray(pcd.normals)


def point2plane(pc1, pc2):
  '''
  pc1, pc2: Nx3
  '''
  
  kdtree = KDTree(pc2)
  pc2_normal = get_normal(pc2)

  distance, index = kdtree.query(pc1, k=1)

  dist = np.linalg.norm(distance, axis=1)
  dist = len(dist[np.where(dist > 0.5)])
  outlier_ratio = dist / distance.shape[0]

  #pc1, pc2, pc2_normal = pc1.T, pc2.T, pc2_normal.T # 3xN
  pc2, pc2_normal = pc2[index.flatten()], pc2_normal[index.flatten()]
  
  b = np.sum(np.hstack((pc2 * pc2_normal, -pc1 * pc2_normal)),axis=1)
  A = pc2_normal[:,2] * pc1[:,1].T - pc2_normal[:,1] * pc1[:,2].T
  A = np.vstack((A, pc2_normal[:,0] * pc1[:,2].T - pc2_normal[:,2] * pc1[:,0].T))
  A = np.vstack((A, pc2_normal[:,1] * pc1[:,0].T - pc2_normal[:,0] * pc1[:,1].T))
  A = np.hstack((A.T, pc2_normal))
  x = np.linalg.pinv(A) @ b
  alpha, beta, gamma, t = x[0], x[1], x[2], x[3:]
  

  thing = ((pc1[:,0]          - gamma * pc1[:,1]             + beta * pc1[:,2]      + t[0] - pc2[:,0])*pc2_normal[:,0] +
          (gamma*pc1[:,0]     + pc1[:,1]                - alpha * pc1[:,2]     + t[1] - pc2[:,1])*pc2_normal[:,1] +
          ((-1)*beta*pc1[:,0] + alpha * pc1[:,1]    + pc1[:,2]  + t[2] - pc2[:,2])*pc2_normal[:,2])

  p2p = np.sum(thing**2)
   # -------------------------
  return p2p, outlier_ratio



def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    #print(":: Apply fast global registration with distance threshold %.3f" \
    #        % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# Parse arguments for is_baseline and input_folder
parser = argparse.ArgumentParser()
parser.add_argument('--is_baseline', type=bool, default=False)
parser.add_argument('--input_folder', type=str)
args = parser.parse_args()


is_baseline = args.is_baseline
input_folder = args.input_folder #"/home/zv/Desktop/lidargen2_samples/waymo_seq/baseline_waymo_seq/"

#is_baseline = False
#input_folder = "/home/zv/Desktop/lidargen2_samples/waymo_updated2/waymo_updated2_lidaronly/"

all_dirs = glob(os.path.join(input_folder, "out*"))


OUTLIER_PERCENT = 0
POINTS_PROCESSED = 0
TOTAL_ENERGY = 0
TOTAL_PERPOINT_ENERGY = 0
TOTAL_CHAMFER = 0

for dir in all_dirs:
    seq = []


    for i in range(0, 5):
        if(is_baseline):
            fn = os.path.join(dir, "pts", f"{str(i)}.pth")
            x = torch.load(fn).detach().cpu().numpy()
            seq.append(x)
        else:
            fn = os.path.join(dir, "raycast", f"{str(i)}.npy")
            x = np.load(fn)
            seq.append(x)

            transforms = np.loadtxt(os.path.join(dir, "poses.txt"))
           

    
    for i in range(0, 4):
        POINTS_PROCESSED = POINTS_PROCESSED + 1
        
        pc_0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seq[i+0]))
        pc_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seq[i+1]))
        

        if(is_baseline):
            pc_0.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pc_1.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            source_down, source_fpfh = preprocess_point_cloud(pc_0, 0.5)
            target_down, target_fpfh = preprocess_point_cloud(pc_1, 0.5)

            result_fast = execute_fast_global_registration(source_down, target_down,
                                               source_fpfh, target_fpfh,
                                               0.5)
            trans_init = result_fast.transformation


            reg_p2l = o3d.pipelines.registration.registration_icp(pc_0, pc_1, 2.0, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane())
            final_transform = reg_p2l.transformation
        else:
            final_transform = transforms[i].reshape(4,4) @ np.linalg.inv(transforms[i].reshape(4,4))
            #print('done')
        
        pc_0.transform(final_transform)


        #o3d.visualization.draw_geometries([pc_0, pc_1])

        energy, m  = point2plane(np.asarray(pc_0.points), np.asarray(pc_1.points))
        chamfer_distance = pc_0.compute_point_cloud_distance(pc_1)

        TOTAL_ENERGY = energy + TOTAL_ENERGY
        TOTAL_PERPOINT_ENERGY = TOTAL_PERPOINT_ENERGY + (energy / np.asarray(pc_0.points).shape[0])

        OUTLIER_PERCENT = m + OUTLIER_PERCENT
        #print(f"Energy is {energy}")

        TOTAL_CHAMFER = np.asarray(chamfer_distance).mean() + TOTAL_CHAMFER
    

    if(POINTS_PROCESSED > 2000):
       break

    
    print(f"{POINTS_PROCESSED} AVG Energy Between frames: {TOTAL_ENERGY / POINTS_PROCESSED}")

    # TODO: Remove this. 
    if((TOTAL_ENERGY / POINTS_PROCESSED) > 2000):
        print('problem')

print("-------------_FINISHED-----------------")
print(f"AVG Energy Between frames:          {TOTAL_ENERGY / POINTS_PROCESSED}")
print(f"AVG PerPoint Energy Between frames: {TOTAL_PERPOINT_ENERGY / POINTS_PROCESSED}")
print(f"AVG Outlier Percent:                {OUTLIER_PERCENT / POINTS_PROCESSED}")
print(f"AVG Chamfer:                        {TOTAL_CHAMFER / POINTS_PROCESSED}")
