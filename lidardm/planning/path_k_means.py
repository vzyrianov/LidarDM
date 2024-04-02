import numpy as np
from lidardm.visualization.dataset2sample import *
from lidardm.planning.utils import * 
import random
from sklearn.cluster import KMeans

def get_n_random(n, dataset):

    random_order = [*range(dataset.__len__())]
    random.shuffle(random_order)
    random_order = random_order[:n]

    all_coords = []

    for idx in random_order:
        record = dataset[idx]
        sample_cpu = sample_to_torch(record)
        sample = sample_to_cuda(sample_cpu)
        plan_pixel = planning_logit_grid_to_waypoints(sample['plan'][0])
        coords = [convert_pixel_meter_torch(p, grid_0=sample['plan'].shape[2], grid_1=sample['plan'].shape[3], pix2m=True) for p in plan_pixel]

        coords_npy = [p.detach().cpu().numpy() for p in coords]
        coords_npy = np.vstack(coords_npy).reshape((-1,))

        all_coords.append(coords_npy)
    
    return np.vstack(all_coords)

def get_clusters(samples, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(samples)
    paths = kmeans.cluster_centers_.reshape((n_clusters, 10, 2))
    return paths