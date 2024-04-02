import numpy as np
import cv2

__all__ = ["cond_to_rgb", "cond_to_rgb_waymo", "flat_visualize", "cond_to_rgb_waymo_finalviz", "attach_lidar_to_map", "visualize_lidardm_map", "visualize_lidar_map_aligned"]


def cond_to_rgb_waymo(cond):
    w = cond.shape[1]
    h = cond.shape[2]
    condn = cond
    
    rgb = np.ones((w*h, 3))
    rgb[np.nonzero(np.reshape(condn[0], (w*h)))[0]] = np.array([0.65, 0.807, 0.89])                     #lane_points
    rgb[np.nonzero(np.reshape(condn[1], (w*h)))[0]] = np.array([0.239, 0.416, 0.89])                     #road_line_points
    rgb[np.nonzero(np.reshape(condn[2], (w*h)))[0]] = np.array([0.416, 0.239, 0.60])                    #road_edge_points
    #rgb[np.nonzero(np.reshape(condn[3], (w*w)))[0]] = np.array([0.416, 0.239, 0.60])                    #stop_sign_points
    rgb[np.nonzero(np.reshape(condn[3], (w*h)))[0]] = np.array([0, 0, 1.0])                             #crosswalk_points
    #rgb[np.nonzero(np.reshape(condn[5], (w*w)))[0]] = np.array([0, 0, 1.0])                             #speed_bump_points
    rgb[np.nonzero(np.reshape(condn[4], (w*h)))[0]] = np.array([1.0, 0, 0.0])                           #driveway_points
    rgb[np.nonzero(np.reshape(condn[5], (w*h)))[0]] = np.array([1.0, 0, 0.0])                           #TYPE_UNKNOWN
    rgb[np.nonzero(np.reshape(condn[6], (w*h)))[0]] = np.array([1.0, 0, 0.0])                           #TYPE_VEHICLE
    rgb[np.nonzero(np.reshape(condn[7], (w*h)))[0]] = np.array([1.0, 0, 0.0])                           #TYPE_PEDESTRIAN
    rgb[np.nonzero(np.reshape(condn[8], (w*h)))[0]] = np.array([1.0, 0, 0.0])                          #TYPE_CYCLIST
    
    rgb = np.reshape(rgb, (w,h,3))

    return rgb

def cond_to_rgb(cond, width=640):
    w = cond.shape[1]
    condn = cond

    rgb = np.ones((w*w, 3))
    rgb[np.nonzero(np.reshape(condn[0], (w*w)))[0]] = np.array([0.65, 0.807, 0.89])
    rgb[np.nonzero(np.reshape(condn[1], (w*w)))[0]] = np.array([0.65, 0.807, 0.89])
    rgb[np.nonzero(np.reshape(condn[2], (w*w)))[0]] = np.array([0.416, 0.239, 0.60])
    rgb[np.nonzero(np.reshape(condn[3], (w*w)))[0]] = np.array([0.416, 0.239, 0.60])
    rgb[np.nonzero(np.reshape(condn[4], (w*w)))[0]] = np.array([0, 0, 1.0])
    rgb[np.nonzero(np.reshape(condn[5], (w*w)))[0]] = np.array([0, 0, 1.0])
    rgb[np.nonzero(np.reshape(condn[6], (w*w)))[0]] = np.array([1.0, 0, 0.0])
    rgb[np.nonzero(np.reshape(condn[7], (w*w)))[0]] = np.array([1.0, 0, 0.0])
    rgb[np.nonzero(np.reshape(condn[8], (w*w)))[0]] = np.array([1.0, 0, 0.0])
    rgb[np.nonzero(np.reshape(condn[9], (w*w)))[0]] = np.array([1.0, 0, 0.0])
    rgb[np.nonzero(np.reshape(condn[10], (w*w)))[0]] = np.array([1.0, 0, 0.0])
    rgb[np.nonzero(np.reshape(condn[11], (w*w)))[0]] = np.array([1.0, 0, 0.0])
    
    rgb = np.reshape(rgb, (w,w,3))

    return rgb

def cond_to_rgb_waymo_finalviz(cond, width=640, plot_origin=True):
  w = width
  condn = cond
      
  rgb = np.ones((w*w, 3))
  rgb[np.nonzero(np.reshape(condn[0], (w*w)))[0]] = np.array([0.85, 0.85, 0.85])                       #lane_points
  rgb[np.nonzero(np.reshape(condn[1], (w*w)))[0]] = np.array([0.75, 0.75, 0.75])                       #road_line_points
  rgb[np.nonzero(np.reshape(condn[2], (w*w)))[0]] = np.array([0.35, 0.35, 0.35])                      #road_edge_points
  #rgb[np.nonzero(np.reshape(condn[3], (w*w)))[0]] = np.array([0.416, 0.239, 0.60])                    #stop_sign_points
  # rgb[np.nonzero(np.reshape(condn[3], (w*w)))[0]] = np.array([0.75, 0.75, 0.75])                             #crosswalk_points
  #rgb[np.nonzero(np.reshape(condn[5], (w*w)))[0]] = np.array([0, 0, 1.0])                             #speed_bump_points
  rgb[np.nonzero(np.reshape(condn[4], (w*w)))[0]] = np.array([0, 1, 0])                          #driveway_points
  rgb[np.nonzero(np.reshape(condn[5], (w*w)))[0]] = np.array([0.65, 0.65, 0.65])                           #TYPE_UNKNOWN
  rgb[np.nonzero(np.reshape(condn[6], (w*w)))[0]] = np.array([0.65, 0.65, 0.65])                           #TYPE_VEHICLE
  rgb[np.nonzero(np.reshape(condn[7], (w*w)))[0]] = np.array([0.65, 0.65, 0.65])                           #TYPE_PEDESTRIAN
  rgb[np.nonzero(np.reshape(condn[8], (w*w)))[0]] = np.array([0.65, 0.65, 0.65])                          #TYPE_CYCLIST

  rgb[np.nonzero(np.reshape(condn[9], (w*w)))[0]] = np.array([0.65, 0.65, 0.65])                           #TYPE_UNKNOWN
  rgb[np.nonzero(np.reshape(condn[10], (w*w)))[0]] = np.array([1.0, 0, 0.0])                           #TYPE_VEHICLE
  rgb[np.nonzero(np.reshape(condn[11], (w*w)))[0]] = np.array([0.0, 0, 1.0])                           #TYPE_PEDESTRIAN
  rgb[np.nonzero(np.reshape(condn[12], (w*w)))[0]] = np.array([0.65, 0.65, 0.65])                          #TYPE_CYCLIST
  rgb = np.reshape(rgb, (w,w,3))
     
  if plot_origin:
    rgb = cv2.circle(rgb, (w//2, w//2), 5, (0,1,0), -1)

  return rgb

def visualize_lidardm_map(lidardm_map, plot_origin=True):
  lidardm_map = np.transpose(lidardm_map, axes=(2, 0, 1))
  lidardm_map = cond_to_rgb_waymo_finalviz(lidardm_map, lidardm_map.shape[1], plot_origin)
  return lidardm_map


def flat_visualize(points, cond_im=None):
    if(len(points.shape) > 3):
        points = points[0]
    image = np.log(np.stack([points.sum(0)]*3, axis=2)+1) / np.log(1+points.shape[0])
    image = 1.0 - (image)

    if(cond_im is None):
        return image
    
    print(cond_im.shape)
    cond = cv2.resize(cond_im, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST_EXACT)

    return 0.798*image + 0.198*cond


def attach_lidar_to_map(map, lidar, width=640, color=[0.22, 0.29, 0.58]):
  w = width
  condn = lidar
      
  rgb = np.reshape(map, (w*w, 3))
  rgb[np.nonzero(np.reshape(lidar, (w*w)))[0]] = np.array(color) #np.array([0.32, 0.39, 0.68]) 
  rgb = np.reshape(rgb, (w,w,3))

  return rgb

def visualize_lidar_map_aligned(bev_map, points, pose):
  from lidardm.core.datasets.utils import voxelize

  lidar_4d = np.ones((points.shape[0], 4))
  lidar_4d[:,:3] = points
  lidar_4d = np.linalg.inv(pose) @ lidar_4d.T
  lidar_3d = (lidar_4d.T)[:,:3]

  lidar = voxelize(lidar_3d, spatial_range=[-47.999, 48.001, -47.999, 48.001, -2.999, 3.001], voxel_size=[0.15, 0.15, 0.15])        
  lidar = lidar.transpose(2, 0, 1)

  lidar = np.rot90(lidar, k=3, axes=(1,2)).copy()
  lidar = np.flip(lidar, 1)
  lidar = np.flip(lidar, 2)

  lidar_im = lidar.sum(0)
  
  final = attach_lidar_to_map(bev_map, (lidar_im>0.5).astype(np.int32))
  return final