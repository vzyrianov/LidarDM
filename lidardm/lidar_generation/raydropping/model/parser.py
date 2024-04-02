import torch

from ..datasets.kitti360_raycast import KITTI360_Raycast
from ..datasets.waymo_raycast import WaymoOpen_Raycast


class Parser():
	def __init__(self,
							 raycast_path: str,
							 raw_path: str,
							 batch_size: int,     
							 workers: int,
							 dataset: str,        
							 shuffle_train=True) -> None:  

		super(Parser, self).__init__()
		
		# if I am training, get the dataset
		self.batch_size = batch_size
		self.workers = workers
		self.shuffle_train = shuffle_train

		train_args = {
			'raycast_path': raycast_path,
			'raw_path': raw_path,
			'split': 'training'
		}

		val_args = {
			'raycast_path': raycast_path,
			'raw_path': raw_path,
			'split': 'validation'
		}

		dataloader_args = {
			"batch_size": self.batch_size,
			"shuffle": self.shuffle_train,
			"num_workers": self.workers,
			"pin_memory": True,
			"drop_last": True
		}

		# Data loading code
		if dataset == 'waymo':
			self.train_dataset = WaymoOpen_Raycast(**train_args)
			self.valid_dataset = WaymoOpen_Raycast(**val_args)
			
		elif dataset == 'kitti360':
			self.train_dataset = KITTI360_Raycast(**train_args)
			self.valid_dataset = KITTI360_Raycast(**val_args)

		else:
			raise ValueError(f'{dataset} not yet supported')

		self.trainloader = torch.utils.data.DataLoader(self.train_dataset, **dataloader_args)
		assert len(self.trainloader) > 0
		self.trainiter = iter(self.trainloader)

		self.validloader = torch.utils.data.DataLoader(self.valid_dataset, **dataloader_args)
		assert len(self.validloader) > 0
		self.validiter = iter(self.validloader)

	def get_train_batch(self):
		return self.trainiter.next()

	def get_train_set(self):
		return self.trainloader

	def get_valid_batch(self):
		return self.validiter.next()

	def get_valid_set(self):
		return self.validloader

	def get_train_size(self):
		return len(self.trainloader)

	def get_valid_size(self):
		return len(self.validloader)
