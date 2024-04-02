from typing import Any, Dict

import lightning as L
from omegaconf import DictConfig
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MetricCollection

from lidardm.core.losses import LossCollection
from lidardm.core.visualizers import VisualizerCollection
from lidardm.visualization.cond2rgb import cond_to_rgb, cond_to_rgb_waymo
import numpy as np

__all__ = ["ModelModule"]


class ModelModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        losses: LossCollection,
        metrics: MetricCollection,
        cfg: DictConfig,
        viz: VisualizerCollection
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.losses = losses
        self.metrics = metrics
        self.visualizations = viz
        self.save_hyperparameters(cfg, ignore=["model", "optimizer", "scheduler", "losses", "metrics"])

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.model(data)

    def training_step(self, data: Dict[str, Any], batch_idx, *args, **kwargs) -> Dict[str, Any]:
        return self._shared_step(data, batch_idx, split="train", on_step=True)

    def validation_step(self, data: Dict[str, Any], batch_idx, *args, **kwargs) -> Dict[str, Any]:
        return self._shared_step(data, batch_idx, split="val", on_step=False)

    def _shared_step(self, data: Dict[str, Any], batch_idx: int, split: str, on_step: bool = False) -> Dict[str, Any]:
        data = self(data)


        loss, losses = self.losses(data)
        self.log(
                f"{split}/losses/FINAL_LOSS",
                loss.detach(),
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )
        self.log_dict(
                {f"{split}/losses/{key}": val.detach() for key, val in losses.items()},
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )


        if split == "train":
            return {"loss": loss}

        if split == "val":
            self.metrics.update(data)

            samples_per_batch = 5
            if(batch_idx < samples_per_batch):
                for logger in self.loggers:
                    if hasattr(logger, "log_image"):

                        for visualizer in self.visualizations.visualizers:
                            viz = self.visualizations.visualizers[visualizer]
                            if(viz.supports_visualization(data)):
                                logger.log_image(viz.output_key, [viz.generate_visualization(data)])

                        if "sample" in data:
                            logger.log_image("sample", [data["sample"][0].sum(0).detach()])
                        if "lidar/recon" in data:
                            recon_binary = (nn.functional.sigmoid(data["lidar/recon"])>0.5).float()
                            logger.log_image("lidar/recon", [recon_binary[0].sum(0).detach()])
                            #logger.log_image("lidar", [data["lidar"].float()[0].sum(0).detach()])

                        if "lidar" in data:
                            logger.log_image("lidar", [data["lidar"].float()[0].sum(0).detach()])

                        if "sample_field" in data:
                            logger.log_image("sample_field", [data["sample_field"][0].sum(0).detach()])
                            if("bev" in data):
                                logger.log_image("sample_field_cond",
                                    [
                                        0.05*(1.0 - np.stack([data["sample_field"][0].sum(0).detach().float().cpu().numpy()]*3, 2)) 
                                        + 0.95 * cond_to_rgb_waymo(data["bev"][0].detach().float().cpu().numpy())
                                    ])
                        if "field/recon" in data:
                            recon_binary = (nn.functional.sigmoid(data["field/recon"]))
                            logger.log_image("field/recon", [recon_binary[0].sum(0).detach()])
                            logger.log_image("field", [data["field"].float()[0].sum(0).detach()])
                        
                        if "bev/recon" in data:
                            recon_map = (nn.functional.sigmoid(data["bev/recon"]) > 0.5)[0].long().detach().cpu().numpy()
                            gt_map = (data["bev"] > 0.5).long()[0].detach().cpu().numpy()

                            if(recon_map.shape[0] == 9):
                                recon_img = cond_to_rgb_waymo(recon_map)
                                gt_img = cond_to_rgb_waymo(gt_map)
                            else:
                                recon_img = cond_to_rgb(recon_map, width=640)
                                gt_img = cond_to_rgb(gt_map, width=640)

                            logger.log_image("bev/recon", [recon_img])
                            logger.log_image("bev", [gt_img])
                    
                    if hasattr(logger, "experiment"):
                        if hasattr(logger.experiment, "add_image"):
                            
                            for visualizer in self.visualizations.visualizers:
                                viz = self.visualizations.visualizers[visualizer]
                                if(viz.supports_visualization(data)):
                                    logger.experiment.add_image(viz.output_key, np.transpose(viz.generate_visualization(data), (2, 0, 1)))


                            if "bev/recon" in data:
                                recon_map = (nn.functional.sigmoid(data["bev/recon"]) > 0.5)[0].long().detach().cpu().numpy()
                                gt_map = (data["bev"] > 0.5).long()[0].detach().cpu().numpy()

                                if(recon_map.shape[0] == 9):
                                    recon_img = cond_to_rgb_waymo(recon_map)
                                    gt_img = cond_to_rgb_waymo(gt_map)
                                else:
                                    recon_img = cond_to_rgb(recon_map, width=640)
                                    gt_img = cond_to_rgb(gt_map, width=640)


                                logger.experiment.add_image("bev/recon", np.transpose(recon_img, (2, 0, 1)))
                                logger.experiment.add_image("bev", np.transpose(gt_img, (2, 0, 1)))

                            if "lidar/recon" in data:
                                recon_binary = (nn.functional.sigmoid(data["lidar/recon"]) > 0.5)[0].long().sum(0).detach().cpu().numpy() / 40.0

                                logger.experiment.add_image("lidar/recon", np.stack([recon_binary, recon_binary, recon_binary], axis=0))

                            if "lidar" in data:
                                recon_binary = (data["lidar"] > 0.5)[0].long().sum(0).detach().cpu().numpy() / 40.0

                                logger.experiment.add_image("lidar", np.stack([recon_binary, recon_binary, recon_binary], axis=0))

                            if "field/recon" in data:
                                recon_binary = (nn.functional.sigmoid(data["field/recon"]))[0].sum(0).detach().cpu().numpy() / 40.0

                                logger.experiment.add_image("field/recon", np.stack([recon_binary, recon_binary, recon_binary], axis=0))

                            if "field" in data:
                                recon_binary = (data["field"])[0].sum(0).detach().cpu().numpy() / 40.0

                                logger.experiment.add_image("field", np.stack([recon_binary, recon_binary, recon_binary], axis=0))

            return {"loss": loss}
    
    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def _log_epoch_metrics(self, split: str) -> None:
        metrics = self.metrics.compute()
        for key, val in metrics.items():
            print(f"{split}/metrics/{key}: {val}")
            self.log(f"{split}/metrics/{key}", val, sync_dist=True)
        self.metrics.reset()

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
