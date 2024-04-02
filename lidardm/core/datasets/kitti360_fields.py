import os
from glob import glob
import gzip
from pathlib import PurePath
from typing import Any, Dict, Tuple

import numpy as np
from torch.utils.data import Dataset
from lidardm.core.datasets.utils import scale_field, unscale_field, load_from_zip_file

from .utils import voxelize_with_value

__all__ = ["KITTI360Fields"]


class KITTI360Fields(Dataset):
	'''
	
	KITTI-360/
	└── data_occupancy_fields/
		├── 2013_05_28_drive_0000_sync/
		│   └── grid/
		│       ├── 18-32.npy.gz
		│       ├── 19-33.npy.gz
		│       └── ...
		└── 2013_05_28_drive_0002_sync/

	'''

	def __init__(
		self,
		root: str,
		split: str,
        spatial_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
		normalization_min = -1.0,
		normalization_max = 1.0
	) -> None:
		self.root = root
		self.split = split
		self.spatial_range = spatial_range
		self.voxel_size = voxel_size
		self.n_min = normalization_min
		self.n_max = normalization_max

		train_seq = ["0002_sync", "0003_sync", "0004_sync", "0005_sync", "0006_sync", "0007_sync", "0009_sync", "0010_sync"]
		val_seq = ["0000_sync"]

		fpaths = glob(os.path.join(root, "data_occupancy_fields", "*", "grid", "*.npy.gz"))

		if split == "train":
			sequence = train_seq
			self.fpaths = [fpath for fpath in fpaths if any(seq in fpath for seq in train_seq)]
		elif split == "val":
			sequence = val_seq
			self.fpaths = [fpath for fpath in fpaths if any(seq in fpath for seq in val_seq)]
		else:
			raise ValueError(f"Invalid split: {split}")

		self.fpaths = sorted(self.fpaths)
	
	def get_field_and_location(self, index:int):
		filename = self.fpaths[index]
		field = load_from_zip_file(filename)
  
		pure_path = PurePath(filename)
		segment_name, field_id = pure_path.parts[-3], pure_path.parts[-1]
		return field, segment_name, field_id

	def __getitem__(self, index: int) -> Dict[str, Any]:
		#index = 0
		field = self.get_field_and_location(index)[0]

		field[:,3] = scale_field(field[:,3], self.n_min, self.n_max)

		volume, intensity_volume = voxelize_with_value(
													field[:,:3],
													field[:,3],
													spatial_range=self.spatial_range,
													voxel_size=self.voxel_size)

		intensity_volume = intensity_volume.transpose(2, 0, 1)
		volume = volume.transpose(2, 0, 1)

		return {
			"field": intensity_volume,
			"lidar": volume, 
			#"field_pts": field,
			#"frames": ()
		}

	def __len__(self) -> int:
		return len(self.fpaths)
