import os
from glob import glob
import gzip
from typing import Any, Dict, Tuple

import numpy as np
from torch.utils.data import Dataset
from pathlib import PurePath
from lidardm.core.datasets.utils import scale_field, unscale_field, load_from_zip_file
from lidardm.core.datasets.utils import encode, decode
from natsort import natsorted

from PIL import Image

from .utils import voxelize_with_value

__all__ = ["WaymoFields"]


class WaymoFields(Dataset):
    '''
    
    Waymo-Field/
    ├── training/
    │	├── segment-.../
    │	│   └── grid/
    │	│       ├── 0-14.npy.gz
    │	│       ├── 0-15.npy.gz
    │	│       └── ...
    │	└── segment-.../
    └── validation/
    '''

    def __init__(
        self,
        root: str,
        root_processed: str,
        split: str,
        spatial_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        normalization_min = -1.0,
        normalization_max = 1.0,
        return_dynamic=False
    ) -> None:
        self.root = root
        self.root_processed = root_processed

        if(split == 'train'):
            self.split = 'training'
        elif(split=='val'):
            self.split='validation'
        #self.split = split
        self.spatial_range = spatial_range
        self.voxel_size = voxel_size
        self.n_min = normalization_min
        self.n_max = normalization_max
        self.return_dynamic = return_dynamic
        #potential_splits = ["training", "testing", "validation"]
        #if split not in potential_splits:
        #	raise ValueError(f"Invalid split: {split}")

        self.fpaths = natsorted(glob(os.path.join(root, self.split, "*", "grid", "*.npy.gz")))
    
    def get_field(self, index:int):
        filename = self.fpaths[index]
        field = load_from_zip_file(filename)
        pure_path = PurePath(filename)

        idx_range = pure_path.parts[-1].split('.')[0].split('-')
        idx0 = int(idx_range[0])
        idx1 = int(idx_range[1])
        center_idx = int(((idx1 - idx0) / 2) + idx0)

        seq_name = pure_path.parts[-3]

        return field, seq_name, center_idx

    def get_dict_for_field(self, field, seq_name, center_idx):
        path_map = os.path.join(self.root_processed, self.split, seq_name, "map", f"{str(center_idx)}.png")
        bev_img = Image.open(path_map)
        bev = decode(bev_img, 13)

        if(self.return_dynamic == False):
            bev = bev[:,:,:9]

        field[:,3] = scale_field(field[:,3], self.n_min, self.n_max)

        volume, intensity_volume = voxelize_with_value(
                                                    field[:,:3],
                                                    field[:,3],
                                                    spatial_range=self.spatial_range,
                                                    voxel_size=self.voxel_size)

        intensity_volume = intensity_volume.transpose(2, 0, 1)
        volume = volume.transpose(2, 0, 1)
        
        bev = np.transpose(bev, axes=(2, 0, 1))
        bev = np.rot90(bev, k=2, axes=(1,2))

        bev = np.rot90(bev, k=1, axes=(1, 2))

        return {
            "field": intensity_volume,
            "lidar": volume,
            "bev": bev.copy().astype(volume.dtype)
        }



    def __getitem__(self, index: int) -> Dict[str, Any]:
        #index = 0
        field, seq_name, center_idx = self.get_field(index)
        return self.get_dict_for_field(field, seq_name, center_idx)
        
        
    def __len__(self) -> int:
        return len(self.fpaths)
