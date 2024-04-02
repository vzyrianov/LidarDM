import os
import numpy as np
import torch
import glob

from pathlib import PurePath
from torch.utils.data import Dataset
from typing import Any, Dict, Tuple
from lidardm.visualization.range_consistency import project_range, KITTI_RANGE_CONFIG

__all__ = ["KITTI360_Raycast"]

class KITTI360_Raycast(Dataset):

	def __init__(self, raycast_path: str, raw_path: str, split: str) -> None:
		self.raycast_path = raycast_path
		self.raw_path = raw_path
		self.split = split

		train_seq = ["0002_sync", "0003_sync", "0004_sync", "0005_sync", "0006_sync", "0007_sync", "0009_sync", "0010_sync"]
		val_seq = ["0000_sync"]

		raycast_paths = glob.glob(os.path.join(raycast_path, "data_raycast", "*", "*.npy"))

		if split == "training":
			fpaths = [fpath for fpath in raycast_paths if any(seq in fpath for seq in train_seq)]
		elif split == "validation":
			fpaths = [fpath for fpath in raycast_paths if any(seq in fpath for seq in val_seq)]
		else:
			raise ValueError(f"Invalid split: {split}")

		self.raycast_paths = sorted(fpaths)
	
	def get_raycast_raw_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
		raycast_filename = self.raycast_paths[index]

		pure_path = PurePath(raycast_filename)
		raycast_id = pure_path.parts[-1][:-4]
		seq_name = pure_path.parts[-2]

		raycast_id = '%010d.bin' % int(raycast_id)
		
		raw_filename = os.path.join(self.raw_path, "data_3d_raw", seq_name, "velodyne_points", "data", raycast_id)
		
		raycast = np.load(raycast_filename)

		if '.bin' in raw_filename:
			raw = np.fromfile(raw_filename, dtype=np.float32).reshape((-1, 4))[:, :3]	
		elif '.npy' in raw_filename:
			raw = np.load(raw_filename)
		return raycast, raw

	def __getitem__(self, index: int) -> Dict[str, Any]:
		raycast, raw = self.get_raycast_raw_pair(index)

		raycast_im = project_range(raycast, **KITTI_RANGE_CONFIG)
		raw_im = project_range(raw, **KITTI_RANGE_CONFIG)
		raw_im_raydrop = (raw_im <= 0)

		raycast_im = torch.as_tensor(np.array(raycast_im)).to(dtype=torch.float)
		raw_im_raydrop = torch.as_tensor(np.array(raw_im_raydrop)).to(dtype=torch.float)

		raycast_im = raycast_im.unsqueeze(0)
		raw_im_raydrop = raw_im_raydrop.unsqueeze(0)

		return raycast_im, raw_im_raydrop

	def __len__(self) -> int:
		return len(self.raycast_paths)