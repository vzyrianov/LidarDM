import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from lidardm.core.losses import LossCollection
from lidardm.core.visualizers import VisualizerCollection

from .modules import DataModule, ModelModule

__all__ = ["instantiate_data", "instantiate_model", "instantiate_trainer"]


def instantiate_data(cfg: DictConfig) -> L.LightningDataModule:
    return DataModule(instantiate(cfg.data.loaders))


def instantiate_model(cfg: DictConfig) -> L.LightningModule:
    model = instantiate(cfg.model)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    losses = LossCollection(**instantiate(cfg.losses))
    metrics = MetricCollection(dict(instantiate(cfg.metrics)), compute_groups=False)

    visualizers = VisualizerCollection(**instantiate(cfg.visualizers))

    return ModelModule(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        losses=losses,
        metrics=metrics,
        cfg=cfg,
        viz=visualizers
    )


def instantiate_trainer(cfg: DictConfig) -> L.Trainer:
    loggers = [instantiate(logger) for logger in cfg.loggers.values()]
    callbacks = [instantiate(callback) for callback in cfg.callbacks.values()]
    return L.Trainer(**instantiate(cfg.trainer), logger=loggers, callbacks=callbacks)
