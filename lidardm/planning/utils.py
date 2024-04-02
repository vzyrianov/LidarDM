import torch
import torch.nn
import numpy as np
from scipy.ndimage import gaussian_filter


def logit_grid_to_waypoint_traj_bank(clusters, logit_grid, apply_softmax=False, blur_cost=None):
    w = logit_grid.shape[1]
    h = logit_grid.shape[2]

    if(apply_softmax):
        logit_grid = torch.nn.functional.softmax(logit_grid.reshape(10, w*h), dim=1).reshape(10, w, h)

    highest_cost = None
    highest_cost_idx = None

    for i in range(0, clusters.shape[0]):
        cluster_coords = [convert_pixel_meter_torch(p, grid_0=logit_grid.shape[1], grid_1=logit_grid.shape[2], pix2m=False) for p in clusters[i]]

        total_cost = 0.0
        total_count = 0.0

        for j in range(len(cluster_coords)):
            if((cluster_coords[j][0] >= 0) and (cluster_coords[j][0] < w) and (cluster_coords[j][1] >= 0) and (cluster_coords[j][1] < h)):
                try:
                    total_cost += logit_grid[j, cluster_coords[j][0], cluster_coords[j][1]]
                    total_count += 1
                except:
                    pass

        avg_cost = total_cost / total_count

        if(highest_cost_idx is None or (avg_cost > highest_cost)):
            highest_cost = avg_cost
            highest_cost_idx = i

    return clusters[highest_cost_idx]


def logit_grid_to_waypoint_traj_bank_pixel(clusters, logit_grid):
    best = logit_grid_to_waypoint_traj_bank(clusters, logit_grid)
    pixel = [convert_pixel_meter_torch(p, grid_0=logit_grid.shape[1], grid_1=logit_grid.shape[2], pix2m=False) for p in best]

    pixel = [p 
                if ((p[0] < logit_grid.shape[1]) and
                    (p[0] >= 0) and
                    (p[1] < logit_grid.shape[2]) and
                    (p[1] >= 0))
                else
                np.array([0,0])
                for p in pixel
            ] 

    return pixel
    


def planning_logit_grid_to_waypoints(logit_grid):
    w = logit_grid.shape[1]
    h = logit_grid.shape[2]

    ws = torch.linspace(0, w-1, steps=w)
    hs = torch.linspace(0, h-1, steps=h)
    ws, hs = torch.meshgrid(ws, hs)
    xy = torch.stack((ws, hs), dim = 0)
    stacked = torch.stack((xy,)*10, dim=1).cuda()

    predicted_plan_softmax = torch.nn.functional.softmax(logit_grid.reshape(10, w*h), dim=1).reshape(10, w, h)


    avg_x = (stacked[0] * predicted_plan_softmax).sum(1).sum(1)
    avg_y = (stacked[1] * predicted_plan_softmax).sum(1).sum(1)

    predicted_waypoints_pixel = [torch.tensor([avg_y[i], avg_x[i]]) for i in range(0, 10)]

    predicted_waypoints_pixel = [
                        (logit_grid[i]==torch.max(logit_grid[i])).nonzero()[0].detach().cpu()
                        for i in range(0, 10)
                        ]

    return predicted_waypoints_pixel


def convert_pixel_meter_torch(value, grid_0, grid_1, pix2m=True, voxel_size=0.15):
    '''
    input:
        - value: the value to convert from/to pixel to/from meter
        - pix2m: True if pixel-> meter, False if meter->pixel
    output:
        - converted value
    '''

    if(torch.is_tensor(value)):
        value_c = value.detach().clone()
    else:
        value_c = value.copy()

    if pix2m: 
        value_c[0] = value_c[0] - (grid_0/2.0)
        value_c[1] = value_c[1] - (grid_1/2.0)
        return value_c*voxel_size
    else: 
        value_c = value_c / voxel_size
        value_c[0] = value_c[0] + (grid_0/2.0)
        value_c[1] = value_c[1] + (grid_1/2.0)
        if(torch.is_tensor(value_c)):
            return value_c.long()
        else:
            return value_c.astype(int)