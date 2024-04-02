import numpy as np

##############################################
#   KITTI-360 DATASET CONFIGS: Velodyne-64   #
##############################################

KITTI_RAYCAST_CONFIG = {
    "azimuth_range": np.array([0, 360]),
    "azimuth_res": 0.09,
    "elevation_range": np.array([-24.8, 2.0]),
    "elevation_beams": 64,
    "max_range": 120,
    "dataset": "kitti360",
    "extrinsic": np.eye(4),
}

KITTI_RAYCAST_LARGE_CONFIG = {
    "azimuth_range": np.array([0, 360]),
    "azimuth_res": 0.09,
    "elevation_range": np.array([-25, 3.0]),
    "elevation_beams": 64,
    "max_range": 120,
    "dataset": "kitti360",
    "extrinsic": np.eye(4),
}

KITTI_RANGE_CONFIG = {
    "range_image_height": 64,
    "range_image_width": 1024,
    "fov_up": 3,
    "fov_down": -25
}

##############################################
#          WAYMO OPEN DATASET CONFIGS        #
##############################################

WAYMO_INCLINATIONS = np.array(
    [4.3682780e-02,  4.0516280e-02,  3.7702467e-02,  3.4711804e-02,
     3.1781308e-02,  2.8802153e-02,  2.6072914e-02,  2.3075983e-02,
     2.0302564e-02,  1.7302759e-02,  1.4139783e-02,  1.1313296e-02,
     8.4363976e-03,  5.5878269e-03,  2.8732109e-03, -2.0284092e-04,
    -3.3035381e-03, -5.9692259e-03, -8.8217165e-03, -1.1916447e-02,
    -1.4747666e-02, -1.7797997e-02, -2.0745473e-02, -2.3857607e-02,
    -2.6894256e-02, -3.0666534e-02, -3.4294609e-02, -3.8212933e-02,
    -4.2541478e-02, -4.6572361e-02, -5.1059261e-02, -5.6004010e-02,
    -6.0479920e-02, -6.5810345e-02, -7.1000807e-02, -7.6193348e-02,
    -8.1705183e-02, -8.7683678e-02, -9.3564264e-02, -9.9754207e-02,
    -1.0630993e-01, -1.1270064e-01, -1.1947037e-01, -1.2650199e-01,
    -1.3343976e-01, -1.4105386e-01, -1.4874573e-01, -1.5633346e-01,
    -1.6443576e-01, -1.7253044e-01, -1.8097438e-01, -1.8901441e-01,
    -1.9778419e-01, -2.0634881e-01, -2.1580023e-01, -2.2529684e-01,
    -2.3453449e-01, -2.4406406e-01, -2.5372660e-01, -2.6313359e-01,
    -2.7382961e-01, -2.8449345e-01, -2.9516894e-01, -3.0624962e-01])

WAYMO_RANGE_CONFIG = {
    "range_image_height": 64,
    "range_image_width": 2048, 
    "fov_up": 2.5,
    "fov_down": -17.5468107035
}

WAYMO_EXTRINSIC = np.array([[1, 0,  0,  1.43000000e+00],
                            [0, 1,  0,  0.00000000e+00],
                            [0, 0,  1,  2.18400000e+00],
                            [0, 0,  0,  1.00000000e+00]])

WAYMO_RAYCAST_CONFIG = {
    "azimuth_range": np.array([0, 360]),
    "azimuth_res": 0.13585,
    "elevation_range": np.array([-17.864249767754103, 2.5935525905175933]),
    "elevation_beams": 64,
    "max_range": 120,
    "dataset": "waymo",
    "extrinsic": WAYMO_EXTRINSIC
}


NUSCENE_RANGE_CONFIG = {
    "range_image_height": 32,
    "range_image_width": 2048, # Approximately equal to (1.4M points/sec) / 32 / 20. 
    "fov_up": 10.0,
    "fov_down": -30.0
}