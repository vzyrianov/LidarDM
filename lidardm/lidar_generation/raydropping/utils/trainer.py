#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import wandb

from ..common.avgmeter import *
from ..common.sync_batchnorm.batchnorm import convert_model
from ..common.warmupLR import *

from ..model.raydropper import RayDropper
from ..model.parser import Parser 

from sklearn.metrics import jaccard_score


def compute_iou(pred, gt):
	return jaccard_score(gt.detach().cpu().numpy().flatten(), pred.detach().cpu().numpy().flatten(), average=None)[1]

def save_img_wandb(raycast, pred_mask, gt_mask, threshold=0.5, split='train'):
	raycast = raycast.detach().clone().cpu().numpy()[0,0]
	pred_mask = pred_mask.detach().clone().cpu().numpy()[0,0]
	gt_mask = gt_mask.detach().clone().cpu().numpy()[0,0]

	# compute raydropped image pair
	pred_raycast = np.clip(raycast.copy(), a_min=0, a_max=None)
	gt_raycast = np.clip(raycast.copy(), a_min=0, a_max=None)
	pred_mask = (pred_mask > threshold).astype(np.uint8)
	gt_mask = (gt_mask > threshold).astype(np.uint8)
	pred_raycast[pred_mask == 1] = 0
	gt_raycast[gt_mask == 1] = 0

	# post-process to display correctly
	pred_raycast = (pred_raycast / np.max(pred_raycast) * 255).astype(np.uint8) 
	gt_raycast = (gt_raycast / np.max(gt_raycast) * 255).astype(np.uint8) 
	pred_mask *= 255
	gt_mask *= 255 

	# create wandb images
	img_pred_raycast = wandb.Image(pred_raycast, caption="pred_raycast")
	img_pred_mask = wandb.Image(pred_mask, caption="pred_mask")
	img_gt_raycast = wandb.Image(gt_raycast, caption="gt_raycast")
	img_gt_mask = wandb.Image(gt_mask, caption="gt_mask")

	wandb.log({f'{split}/raycast': [img_pred_raycast, img_gt_raycast]})
	wandb.log({f'{split}/mask': [img_pred_mask, img_gt_mask]})

	return img_pred_raycast, img_pred_mask, img_gt_raycast, img_gt_mask

class Trainer():
	def __init__(self, arch_config, dataset, raycast_path, raw_path, logdir, path=None):
		
		# parameters
		self.arch_config = arch_config
		self.log = logdir
		self.path = path

		# get the data
		self.parser = Parser(raycast_path=raycast_path,
												 raw_path=raw_path,
												 batch_size=self.arch_config["train"]["batch_size"],
												 workers=self.arch_config["train"]["workers"],
												 shuffle_train=True)
		
		self.loss_w = torch.Tensor([0.5])

		# concatenate the encoder and the head
		with torch.no_grad():
			self.model = RayDropper(self.arch_config, self.path)

		# GPU?
		self.gpu = False
		self.multi_gpu = False
		self.n_gpus = 0
		self.model_single = self.model
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print("Training in device: ", self.device)

		if torch.cuda.is_available() and torch.cuda.device_count() > 0:
			cudnn.benchmark = True
			cudnn.fastest = True
			self.gpu = True
			self.n_gpus = 1
			self.model.cuda()

		if torch.cuda.is_available() and torch.cuda.device_count() > 1:
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			self.model = nn.DataParallel(self.model)   # spread in gpus
			self.model = convert_model(self.model).cuda()  # sync batchnorm
			self.model_single = self.model.module  # single model to get weight names
			self.multi_gpu = True
			self.n_gpus = torch.cuda.device_count()

		# loss
		self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.loss_w).to(self.device)
		
		# loss as dataparallel too (more images in batch)
		if self.n_gpus > 1:
			self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus

		# optimizer
		self.lr_group_names = []
		self.train_dicts = []

		if self.arch_config["backbone"]["train"]:
			self.lr_group_names.append("backbone_lr")
			self.train_dicts.append(
					{'params': self.model_single.backbone.parameters()})
			
		if self.arch_config["decoder"]["train"]:
			self.lr_group_names.append("decoder_lr")
			self.train_dicts.append(
					{'params': self.model_single.decoder.parameters()})
			
		if self.arch_config["head"]["train"]:
			self.lr_group_names.append("head_lr")
			self.train_dicts.append({'params': self.model_single.head.parameters()})

		# Use SGD optimizer to train
		self.optimizer = optim.SGD(self.train_dicts,
															 lr=self.arch_config["train"]["lr"],
															 momentum=self.arch_config["train"]["momentum"],
															 weight_decay=self.arch_config["train"]["w_decay"])

		# Use warmup learning rate
		# post decay and step sizes come in epochs and we want it in steps
		steps_per_epoch = self.parser.get_train_size()
		up_steps = int(self.arch_config["train"]["wup_epochs"] * steps_per_epoch)
		final_decay = self.arch_config["train"]["lr_decay"] ** (1/steps_per_epoch)
		self.scheduler = warmupLR(optimizer=self.optimizer,
															lr=self.arch_config["train"]["lr"],
															warmup_steps=up_steps,
															momentum=self.arch_config["train"]["momentum"],
															decay=final_decay)

		wandb.init(project=f'raydrop-{dataset}')

	def train(self):
		# accuracy and mAP stuff
		best_train_iou = 0.0
		best_val_iou = 0.0

		self.evaluator = None

		# train for n epochs
		for epoch in range(self.arch_config["train"]["max_epochs"]):

			# train for 1 epoch
			iou = self.train_epoch(train_loader=self.parser.get_train_set(),
														 model=self.model,
														 criterion=self.criterion,
														 optimizer=self.optimizer,
														 epoch=epoch,
														 scheduler=self.scheduler)

			# remember best iou and save checkpoint
			if iou >= best_train_iou:
				print("Best mean iou in training set so far, save model!")
				best_train_iou = iou
				self.model_single.save_checkpoint(self.log, suffix=f'_{epoch}_train.pt')

			if epoch % self.arch_config["train"]["report_epoch"] == 0:
				# evaluate on validation set
				print("*" * 80)
				iou = self.validate(val_loader=self.parser.get_valid_set(),
														model=self.model,
														criterion=self.criterion)

				# remember best iou and save checkpoint
				if iou > best_val_iou:
					print("Best mean iou in validation so far, save model!")
					print("*" * 80)
					best_val_iou = iou

					# save the weights!
					self.model_single.save_checkpoint(self.log, suffix=f'_{epoch}_val.pt')

				print("*" * 80)
		print('Finished Training')

		return

	def train_epoch(self, train_loader, model, criterion, optimizer, epoch, scheduler):
		losses = AverageMeter()
		iou = AverageMeter()

		# empty the cache to train now
		if self.gpu:
			torch.cuda.empty_cache()

		# switch to train mode
		model.train()

		for i, (raycast_im, gt_mask) in enumerate(train_loader):

			if not self.multi_gpu and self.gpu:
				raycast_im = raycast_im.cuda()
			if self.gpu:
				gt_mask = gt_mask.cuda(non_blocking=True).float()

			output = model(raycast_im)
			loss = criterion(output, gt_mask)

			# compute gradient and do SGD step
			optimizer.zero_grad()
			if self.n_gpus > 1:
				idx = torch.ones(self.n_gpus).cuda()
				loss.backward(idx)
			else:
				loss.backward()
			optimizer.step()

			# measure accuracy and record loss
			loss = loss.mean()
			with torch.no_grad():
				pred_mask = torch.sigmoid(output) >= 0.5
				current_iou = compute_iou(pred_mask, gt_mask)
				
			losses.update(loss.item(), raycast_im.size(0))
			iou.update(current_iou.item(), raycast_im.size(0))

			# update wandb
			if i % self.arch_config["train"]["report_batch_image"] == 0:
					save_img_wandb(raycast_im, pred_mask, gt_mask)
			
			if i % self.arch_config["train"]["report_batch"] == 0:          
				wandb.log({
						'epoch': epoch, 
						'lr_sgd': optimizer.param_groups[0]['lr'],
						'batches': i,
						'train_iou': iou.val,
						'train_loss': losses.val,
				})
			# step scheduler
			scheduler.step()

		return iou.avg

	def validate(self, val_loader, model, criterion):
		losses = AverageMeter()
		iou = AverageMeter()

		# switch to evaluate mode
		model.eval()

		# empty the cache to infer in high res
		if self.gpu:
			torch.cuda.empty_cache()

		with torch.no_grad():

			for i, (raycast_im, gt_mask) in enumerate(val_loader):
				if not self.multi_gpu and self.gpu:
					raycast_im = raycast_im.cuda()
				if self.gpu:
					gt_mask = gt_mask.cuda(non_blocking=True).float()

				# compute output
				output = model(raycast_im)
				loss = criterion(output, gt_mask)

				# measure accuracy and record loss
				losses.update(loss.mean().item(), raycast_im.size(0))

				pred_mask = torch.sigmoid(output) >= 0.5
				current_iou = compute_iou(pred_mask, gt_mask)
				iou.update(current_iou, raycast_im.size(0))

				if i % self.arch_config["val"]["report_batch_image"] == 0:
					save_img_wandb(raycast_im, pred_mask, gt_mask)
					
			wandb.log({
					'val_iou': iou.val,
					'val_loss': losses.val,
			})
			
		return iou.avg
