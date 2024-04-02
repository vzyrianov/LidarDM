import os
from glob import glob
from typing import Any, Dict, Tuple
from lidardm.core.datasets.utils import encode, decode

import numpy as np
from torch.utils.data import Dataset
from pathlib import PurePath

from pathlib import PurePath
from .utils import voxelize
from natsort import natsorted

from PIL import Image

__all__ = ["WaymoSeq"]

class WaymoSeq(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        spatial_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float]
    ) -> None:
        self.spatial_range = spatial_range
        self.voxel_size = voxel_size
        self.root = root
        self.fpaths = glob(os.path.join(self.root, "training" if split == 'train' else 'validation', "*.tfrecord", "lidar_scan_raw", "*.npy"))
        self.return_dynamic = True
        
        self.fpaths = sorted(self.fpaths)

        fpaths_copy = self.fpaths.copy()

        sequence_paths = glob(os.path.join(self.root,  "training" if split == 'train' else 'validation', "*.tfrecord"))
        sequence = [PurePath(x).parts[-1] for x in sequence_paths]


        for seq_name in sequence:
            subsequence = natsorted([p for p in fpaths_copy if (seq_name in p)])
            for j in range(1, 5):
                self.fpaths.remove(subsequence[-j])



    def load_from_filename(self, filename: str):
        lidar = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        lidar = lidar[:,:3]

        return lidar
    
        
    def increment_filename(self, filename, inc):
        s = filename

        pure_path = PurePath(filename)

        new_path = pure_path.with_stem(str(int(pure_path.parts[-1].split('.')[0])+inc))

        
        return str(new_path.as_posix())



    def __getitem__(self, index: int) -> Dict[str, Any]:
        lidar_filenames = [self.fpaths[index]]
        for i in range(0, 4):
            lidar_filenames.append(self.increment_filename(lidar_filenames[0], i+1))

        
        lidars = [np.load(x) for x in lidar_filenames]
        voxelized = [voxelize(lidar[:, :3], spatial_range=self.spatial_range, voxel_size=self.voxel_size).transpose(2,0,1) for lidar in lidars]

        map_filename = lidar_filenames[2].replace("lidar_scan_raw", "map").replace(".npy", ".png")
        bev = decode(Image.open(map_filename), 13)
        
        bev[:,:,5:9] = ((bev[:,:,5:9] + bev[:,:,9:]) > 0).astype(bev.dtype)
        bev = bev[:,:,:9]

        voxelized = [np.rot90(v, k=3, axes=(1,2)) for v in voxelized]

        bev = np.transpose(bev, axes=(2, 0, 1))
        bev = np.rot90(bev, k=2, axes=(1,2))

        return {
            "lidar": voxelized[2].copy(),
            "lidar_seq": np.stack(voxelized, axis=0),
            "bev": bev.copy().astype(voxelized[0].dtype)
        }



    def __len__(self) -> int:
        return len(self.fpaths)  # if self.split == "val" else 10
