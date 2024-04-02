import concurrent.futures
from functools import partial
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

__all__ = ["MaximumMeanDiscrepancy"]

def debug_viz(s1, s2):
    import matplotlib.pyplot as plt 
    f, axarr = plt.subplots(2)
    axarr[0].imshow(s1)
    axarr[1].imshow(s2)
    f.show()


class MaximumMeanDiscrepancy(Metric):
    def __init__(self) -> None:
        super().__init__(dist_sync_on_step=False)

        self.add_state("gt_set", default=[], dist_reduce_fx="cat")
        self.add_state("gen_set", default=[], dist_reduce_fx="cat")

    def update(self, data: Dict[str, Any]) -> None:
        #debug_viz(self._flatten_samples(data["lidar"]).sum(0), self._flatten_samples(data["sample"]).sum(0))

        self.gen_set.append(self._flatten_samples(data["sample"]))
        self.gt_set.append(self._flatten_samples(data["lidar"]))

    def _flatten_samples(self, sample):
        return F.adaptive_avg_pool3d(sample, (1, 100, 100)).squeeze(1)

    def compute(self):
        catted1 = dim_zero_cat(self.gt_set)
        dist1_unnormalized = catted1.float().cpu().numpy()
        batch_size1 = dist1_unnormalized.shape[0]
        dist1_unnormalized_list = [dist1_unnormalized[i].flatten() for i in range(batch_size1)]

        catted2 = dim_zero_cat(self.gen_set)
        dist2_unnormalized = catted2.float().cpu().numpy()
        batch_size2 = dist2_unnormalized.shape[0]
        dist2_unnormalized_list = [dist2_unnormalized[i].flatten() for i in range(batch_size2)]

        mmd = compute_mmd(dist1_unnormalized_list, dist2_unnormalized_list, gaussian, is_hist=True)
        return mmd


def gaussian(x, y, sigma=0.5):
    support_size = max(len(x), len(y))

    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    # TODO: Calculate empirical sigma by fitting dist to gaussian
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0

    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dist in executor.map(
                kernel_parallel_worker, [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]
            ):
                d += dist

    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """MMD between two samples"""
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    return (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    )
