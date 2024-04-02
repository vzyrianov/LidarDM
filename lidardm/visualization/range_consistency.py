	
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["range_consistency", "range_consistency_with_dataset", "unproject_range", 
           "project_range"]

def range_consistency_with_dataset(
        points: np.ndarray,
        dataset_name: str) -> np.ndarray:

    from lidardm.lidar_generation.scene_composition.utils.raycast_configs import KITTI_RANGE_CONFIG, NUSCENE_RANGE_CONFIG
    if(dataset_name == 'kitti'):
        return range_consistency(points, **KITTI_RANGE_CONFIG)
    elif(dataset_name == 'nuscenes'):
        return range_consistency(points, **NUSCENE_RANGE_CONFIG)

def range_consistency(
        points: np.ndarray,
        range_image_height : int,
        range_image_width: int,
        fov_up: float ,
        fov_down: float) -> np.ndarray:
    
    range_image = project_range(points, 
                                range_image_height=range_image_height, 
                                range_image_width=range_image_width, 
                                fov_up=fov_up, 
                                fov_down=fov_down)
    
    new_points = unproject_range(range_image, 
                                 range_image_height=range_image_height, 
                                 range_image_width=range_image_width, 
                                 fov_up=fov_up, 
                                 fov_down=fov_down)

    return new_points

def project_range(
        points: np.ndarray,
        range_image_height : int,
        range_image_width: int,
        fov_up: float ,
        fov_down: float) -> np.ndarray:
    
    laser_scan = LaserScan(H=range_image_height, W=range_image_width, fov_up=fov_up, fov_down=fov_down)
    laser_scan.set_points(points)
    laser_scan.do_range_projection()

    return laser_scan.proj_range

def unproject_range(
        range_image: np.ndarray,
        range_image_height : int,
        range_image_width: int,
        fov_up: float,
        fov_down: float):
    
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    x, y = np.meshgrid(np.arange(0, range_image_width), np.arange(0, range_image_height))

    x = x.astype(float)
    y = y.astype(float)

    x *= 1/range_image_width
    y *= 1/range_image_height

    yaw = np.pi*(x * 2 - 1)
    pitch = (1.0 - y)*fov - abs(fov_down)

    yaw = yaw.flatten()
    pitch = pitch.flatten()
    depth = range_image.flatten()

    pts = np.zeros((len(yaw), 3))
    pts[:, 0] =  np.cos(yaw) * np.cos(pitch) * depth
    pts[:, 1] =  -np.sin(yaw) * np.cos(pitch) * depth
    pts[:, 2] =  np.sin(pitch) * depth

    mask = np.logical_and(depth>0.0, depth < 90.0)
    xyz = pts[mask, :]
    return xyz

'''
    Adapted fom semantic-kitti-api project.  https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/laserscan.py
'''
class LaserScan:
    def __init__(self, project=False, H=32, W=1024, fov_up=10.0, fov_down=-30.0):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros(
            (0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros(
            (0, 1), dtype=np.float32)    # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)       # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points    # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W                              # in [0.0, W]
        proj_y *= self.proj_H                              # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)