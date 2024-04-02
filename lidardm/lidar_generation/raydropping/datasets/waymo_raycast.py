import os
import numpy as np
import torch
import glob

from pathlib import PurePath
from torch.utils.data import Dataset
from typing import Any, Dict, Tuple

from lidardm.visualization.range_consistency import project_range, WAYMO_RANGE_CONFIG, WAYMO_EXTRINSIC
from lidardm.lidar_generation.scene_composition.utils.utils import transform_points_to_pose

__all__ = ["WaymoOpen_Raycast"]

class WaymoOpen_Raycast(Dataset):

	def __init__(self, raycast_path: str, raw_path: str, split: str) -> None:
		self.raycast_path = raycast_path
		self.raw_path = raw_path
		self.split = split
	
		potential_splits = ["training", "testing", "validation"]
		if split not in potential_splits:
			raise ValueError(f"Invalid split: {split}")

		self.raycast_paths = sorted(glob.glob(os.path.join(raycast_path, split, "*", "raycast", "*.npy")))
	
	def get_raycast_raw_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
		raycast_filename = self.raycast_paths[index]

		pure_path = PurePath(raycast_filename)
		raycast_id = pure_path.parts[-1]
		seq_name = pure_path.parts[-3]
		
		raw_filename = os.path.join(self.raw_path, self.split, seq_name, "raw_lidar_scans", raycast_id)
		
		raycast = np.load(raycast_filename)
		raw = np.load(raw_filename)
		return raycast, raw

	def __getitem__(self, index: int) -> Dict[str, Any]:
		raycast, raw = self.get_raycast_raw_pair(index)

		raycast_trans = transform_points_to_pose(raycast, np.linalg.inv(WAYMO_EXTRINSIC))
		raw_trans = transform_points_to_pose(raw, np.linalg.inv(WAYMO_EXTRINSIC))

		raycast_im = project_range(raycast_trans, **WAYMO_RANGE_CONFIG)
		raw_im = project_range(raw_trans, **WAYMO_RANGE_CONFIG)
		raw_im_raydrop = (raw_im <= 0)

		raycast_im = torch.as_tensor(np.array(raycast_im)).to(dtype=torch.float)
		raw_im_raydrop = torch.as_tensor(np.array(raw_im_raydrop)).to(dtype=torch.float)

		raycast_im = raycast_im.unsqueeze(0)
		raw_im_raydrop = raw_im_raydrop.unsqueeze(0)

		return raycast_im, raw_im_raydrop

	def __len__(self) -> int:
		return len(self.raycast_paths)